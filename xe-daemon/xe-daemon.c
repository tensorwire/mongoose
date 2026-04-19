// xe-daemon: persistent C process that owns the Intel Xe GPU via Level Zero.
// Go communicates via shared memory command ring + Unix socket for control.
// Exists because Intel's IGC JIT compiler crashes in Go's address space (~50%)
// but works 100% in standalone C binaries. This daemon IS the C binary.
//
// Protocol:
//   1. Go starts xe-daemon as a child process
//   2. Daemon creates L0 context, allocates shared memory, listens on Unix socket
//   3. Go mmaps the shared memory region (GPU-accessible from both processes)
//   4. Go sends commands via socket: "dispatch rmsnorm <args>" / "alloc <size>" / "sync"
//   5. Daemon executes on Xe GPU, Go reads results from shared memory
//
// Build: gcc -O2 -o xe-daemon xe-daemon.c -lze_loader -lm -lpthread
// Usage: xe-daemon /tmp/mongoose-xe.sock

#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

// L0 globals
static ze_driver_handle_t  g_driver  = NULL;
static ze_device_handle_t  g_device  = NULL;
static ze_context_handle_t g_context = NULL;
static ze_command_list_handle_t g_cmdlist = NULL;
static char g_device_name[256] = {0};
static uint64_t g_mem_size = 0;

// SPIR-V kernel handles
#define MAX_KERNELS 16
static ze_module_handle_t g_modules[MAX_KERNELS];
static ze_kernel_handle_t g_kernels[MAX_KERNELS];
static char g_kernel_names[MAX_KERNELS][64];
static int g_kernel_count = 0;

// Shared memory allocations tracked for cleanup
#define MAX_ALLOCS 256
static void* g_allocs[MAX_ALLOCS];
static size_t g_alloc_sizes[MAX_ALLOCS];
static int g_alloc_count = 0;

// Split shared memory arena — zero-copy data exchange with Go.
// Layout:
//   [0 .. HALF)           — GO REGION: Go writes logits + targets
//   [HALF .. HALF+4096)   — GUARD: 4KB dead zone (never touched)
//   [HALF+4096 .. SIZE)   — XE REGION: Xe writes losses + gradients
//
// Go passes the memfd file descriptor as argv[2]. Both processes mmap it.
// No filesystem, no POSIX shm, no names. Just an anonymous fd + two mmaps.
#define ARENA_SIZE      (256 * 1024 * 1024)
#define ARENA_HALF      (ARENA_SIZE / 2)
#define ARENA_GUARD     4096
#define ARENA_XE_START  (ARENA_HALF + ARENA_GUARD)

static int g_arena_fd = -1;
static void* g_arena_base = NULL;

static volatile int g_running = 1;

static void sighandler(int sig) {
    g_running = 0;
}

static int init_level_zero() {
    ze_result_t r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[xe-daemon] zeInit failed: %d\n", r);
        return -1;
    }

    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, NULL);
    if (driverCount == 0) return -2;

    ze_driver_handle_t drivers[4];
    zeDriverGet(&driverCount, drivers);

    for (uint32_t d = 0; d < driverCount; d++) {
        uint32_t devCount = 0;
        zeDeviceGet(drivers[d], &devCount, NULL);
        if (devCount == 0) continue;

        ze_device_handle_t devices[8];
        zeDeviceGet(drivers[d], &devCount, devices);

        for (uint32_t i = 0; i < devCount; i++) {
            ze_device_properties_t props = {.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            zeDeviceGetProperties(devices[i], &props);

            if (props.type == ZE_DEVICE_TYPE_GPU && props.vendorId == 0x8086) {
                g_driver = drivers[d];
                g_device = devices[i];
                strncpy(g_device_name, props.name, 255);
                g_mem_size = 0;

                ze_device_memory_properties_t memProps[4];
                uint32_t memCount = 4;
                zeDeviceGetMemoryProperties(g_device, &memCount, memProps);
                for (uint32_t m = 0; m < memCount; m++)
                    g_mem_size += memProps[m].totalSize;

                goto found;
            }
        }
    }
    return -3;

found:;
    ze_context_desc_t ctxDesc = {.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    r = zeContextCreate(g_driver, &ctxDesc, &g_context);
    if (r != ZE_RESULT_SUCCESS) return -4;

    // Find compute queue ordinal
    uint32_t qgCount = 0;
    zeDeviceGetCommandQueueGroupProperties(g_device, &qgCount, NULL);
    ze_command_queue_group_properties_t qgProps[8];
    for (uint32_t i = 0; i < qgCount && i < 8; i++)
        qgProps[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    zeDeviceGetCommandQueueGroupProperties(g_device, &qgCount, qgProps);

    uint32_t computeOrd = 0;
    for (uint32_t i = 0; i < qgCount; i++) {
        if (qgProps[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            computeOrd = i;
            break;
        }
    }

    ze_command_queue_desc_t qDesc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal = computeOrd,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    r = zeCommandListCreateImmediate(g_context, g_device, &qDesc, &g_cmdlist);
    if (r != ZE_RESULT_SUCCESS) return -5;

    return 0;
}

// Map the arena from an inherited memfd file descriptor.
// Go creates the memfd, passes fd number as argv[2].
static int init_arena_from_fd(int fd) {
    g_arena_fd = fd;
    g_arena_base = mmap(NULL, ARENA_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (g_arena_base == MAP_FAILED) {
        fprintf(stderr, "[xe-daemon] arena mmap failed: %s\n", strerror(errno));
        g_arena_base = NULL;
        return -1;
    }
    // Protect the guard page — any accidental cross-boundary access segfaults
    mprotect((char*)g_arena_base + ARENA_HALF, ARENA_GUARD, PROT_NONE);
    fprintf(stderr, "[xe-daemon] arena mapped: fd=%d, %d MB (go: 0-%dMB, guard: 4KB, xe: %dMB-%dMB)\n",
            fd, ARENA_SIZE / (1024*1024),
            ARENA_HALF / (1024*1024),
            ARENA_XE_START / (1024*1024), ARENA_SIZE / (1024*1024));
    return 0;
}

// Dispatch fused cross-entropy on the split arena.
// logits + targets are in GO REGION (offsets < ARENA_HALF).
// losses + grad are written to XE REGION (offsets >= ARENA_XE_START).
static int dispatch_cross_entropy(int kidx, uint32_t logits_off, uint32_t targets_off,
                                   uint32_t losses_off, uint32_t grad_off,
                                   uint32_t n_pos, uint32_t vocab_size, float inv_n) {
    if (kidx < 0 || kidx >= g_kernel_count) return -1;
    if (!g_arena_base) return -2;

    // Validate: inputs in Go region, outputs in Xe region
    if (logits_off >= ARENA_HALF || targets_off >= ARENA_HALF) return -3;
    if (losses_off < ARENA_XE_START || grad_off < ARENA_XE_START) return -4;

    void* logits_ptr  = (char*)g_arena_base + logits_off;
    void* targets_ptr = (char*)g_arena_base + targets_off;
    void* losses_ptr  = (char*)g_arena_base + losses_off;
    void* grad_ptr    = (char*)g_arena_base + grad_off;

    ze_kernel_handle_t k = g_kernels[kidx];
    zeKernelSetGroupSize(k, 256, 1, 1);
    zeKernelSetArgumentValue(k, 0, sizeof(void*), &logits_ptr);
    zeKernelSetArgumentValue(k, 1, sizeof(void*), &targets_ptr);
    zeKernelSetArgumentValue(k, 2, sizeof(void*), &losses_ptr);
    zeKernelSetArgumentValue(k, 3, sizeof(void*), &grad_ptr);
    zeKernelSetArgumentValue(k, 4, sizeof(uint32_t), &vocab_size);
    zeKernelSetArgumentValue(k, 5, sizeof(float), &inv_n);

    ze_group_count_t disp = {n_pos, 1, 1};
    zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
    return 0;
}

// Allocate from L0 shared memory (host+device accessible)
static void* xe_alloc(size_t bytes) {
    void* ptr = NULL;
    ze_device_mem_alloc_desc_t devDesc = {.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t hostDesc = {.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    ze_result_t r = zeMemAllocShared(g_context, &devDesc, &hostDesc, bytes, 64, g_device, &ptr);
    if (r != ZE_RESULT_SUCCESS) return NULL;

    if (g_alloc_count < MAX_ALLOCS) {
        g_allocs[g_alloc_count] = ptr;
        g_alloc_sizes[g_alloc_count] = bytes;
        g_alloc_count++;
    }
    return ptr;
}

static void xe_free_ptr(void* ptr) {
    zeMemFree(g_context, ptr);
    for (int i = 0; i < g_alloc_count; i++) {
        if (g_allocs[i] == ptr) {
            g_allocs[i] = g_allocs[g_alloc_count - 1];
            g_alloc_sizes[i] = g_alloc_sizes[g_alloc_count - 1];
            g_alloc_count--;
            break;
        }
    }
}

static int load_spirv(const char* path, const char* entry) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);

    ze_module_desc_t modDesc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .format = ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = sz,
        .pInputModule = buf
    };
    ze_module_handle_t mod = NULL;
    ze_module_build_log_handle_t buildLog = NULL;
    ze_result_t r = zeModuleCreate(g_context, g_device, &modDesc, &mod, &buildLog);
    free(buf);

    if (r != ZE_RESULT_SUCCESS) {
        if (buildLog) {
            size_t logSz = 0;
            zeModuleBuildLogGetString(buildLog, &logSz, NULL);
            char* log = malloc(logSz + 1);
            zeModuleBuildLogGetString(buildLog, &logSz, log);
            fprintf(stderr, "[xe-daemon] SPIR-V build: %s\n", log);
            free(log);
            zeModuleBuildLogDestroy(buildLog);
        }
        return -2;
    }
    if (buildLog) zeModuleBuildLogDestroy(buildLog);

    ze_kernel_desc_t kDesc = {.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC, .pKernelName = entry};
    ze_kernel_handle_t kern = NULL;
    r = zeKernelCreate(mod, &kDesc, &kern);
    if (r != ZE_RESULT_SUCCESS) {
        zeModuleDestroy(mod);
        return -3;
    }

    int idx = g_kernel_count++;
    g_modules[idx] = mod;
    g_kernels[idx] = kern;
    strncpy(g_kernel_names[idx], entry, 63);
    return idx;
}

static void xe_sync() {
    zeCommandListHostSynchronize(g_cmdlist, UINT64_MAX);
}

// Dispatch RMSNorm kernel
static int dispatch_rmsnorm(int kidx, void* x, void* out, void* weight,
                             uint32_t dim, uint32_t seqLen, float eps) {
    if (kidx < 0 || kidx >= g_kernel_count) return -1;
    ze_kernel_handle_t k = g_kernels[kidx];
    zeKernelSetGroupSize(k, 256, 1, 1);
    zeKernelSetArgumentValue(k, 0, sizeof(void*), &x);
    zeKernelSetArgumentValue(k, 1, sizeof(void*), &out);
    zeKernelSetArgumentValue(k, 2, sizeof(void*), &weight);
    zeKernelSetArgumentValue(k, 3, sizeof(uint32_t), &dim);
    zeKernelSetArgumentValue(k, 4, sizeof(uint32_t), &seqLen);
    zeKernelSetArgumentValue(k, 5, sizeof(float), &eps);
    ze_group_count_t disp = {seqLen, 1, 1};
    zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
    return 0;
}

// Dispatch SiLU-gate-mul kernel
static int dispatch_silu(int kidx, void* gate, void* up, void* out, uint32_t n) {
    if (kidx < 0 || kidx >= g_kernel_count) return -1;
    ze_kernel_handle_t k = g_kernels[kidx];
    zeKernelSetGroupSize(k, 256, 1, 1);
    zeKernelSetArgumentValue(k, 0, sizeof(void*), &gate);
    zeKernelSetArgumentValue(k, 1, sizeof(void*), &up);
    zeKernelSetArgumentValue(k, 2, sizeof(void*), &out);
    zeKernelSetArgumentValue(k, 3, sizeof(uint32_t), &n);
    uint32_t groups = (n + 255) / 256;
    ze_group_count_t disp = {groups, 1, 1};
    zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
    return 0;
}

// Dispatch add_inplace kernel
static int dispatch_add(int kidx, void* a, void* b, uint32_t n) {
    if (kidx < 0 || kidx >= g_kernel_count) return -1;
    ze_kernel_handle_t k = g_kernels[kidx];
    zeKernelSetGroupSize(k, 256, 1, 1);
    zeKernelSetArgumentValue(k, 0, sizeof(void*), &a);
    zeKernelSetArgumentValue(k, 1, sizeof(void*), &b);
    zeKernelSetArgumentValue(k, 2, sizeof(uint32_t), &n);
    uint32_t groups = (n + 255) / 256;
    ze_group_count_t disp = {groups, 1, 1};
    zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
    return 0;
}

// Command handler — reads one line from socket, executes, responds.
static void handle_cmd(int fd, char* cmd) {
    char resp[512];

    if (strncmp(cmd, "info", 4) == 0) {
        snprintf(resp, sizeof(resp), "OK %s %lu %d\n",
                 g_device_name, (unsigned long)g_mem_size, g_kernel_count);
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "load ", 5) == 0) {
        // load <path> <entry>
        char path[256], entry[64];
        if (sscanf(cmd + 5, "%255s %63s", path, entry) == 2) {
            int idx = load_spirv(path, entry);
            snprintf(resp, sizeof(resp), "OK %d\n", idx);
        } else {
            snprintf(resp, sizeof(resp), "ERR bad args\n");
        }
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "alloc ", 6) == 0) {
        // alloc <bytes> → returns pointer as hex
        size_t bytes = 0;
        sscanf(cmd + 6, "%zu", &bytes);
        void* ptr = xe_alloc(bytes);
        snprintf(resp, sizeof(resp), "OK %p\n", ptr);
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "free ", 5) == 0) {
        void* ptr = NULL;
        sscanf(cmd + 5, "%p", &ptr);
        if (ptr) xe_free_ptr(ptr);
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "rmsnorm ", 8) == 0) {
        // rmsnorm <kidx> <x_ptr> <out_ptr> <w_ptr> <dim> <seqLen> <eps>
        int kidx;
        void *x, *out, *w;
        uint32_t dim, seqLen;
        float eps;
        sscanf(cmd + 8, "%d %p %p %p %u %u %f", &kidx, &x, &out, &w, &dim, &seqLen, &eps);
        dispatch_rmsnorm(kidx, x, out, w, dim, seqLen, eps);
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "silu ", 5) == 0) {
        int kidx;
        void *gate, *up, *out;
        uint32_t n;
        sscanf(cmd + 5, "%d %p %p %p %u", &kidx, &gate, &up, &out, &n);
        dispatch_silu(kidx, gate, up, out, n);
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "add ", 4) == 0) {
        int kidx;
        void *a, *b;
        uint32_t n;
        sscanf(cmd + 4, "%d %p %p %u", &kidx, &a, &b, &n);
        dispatch_add(kidx, a, b, n);
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "arena", 5) == 0) {
        if (g_arena_base) {
            snprintf(resp, sizeof(resp), "OK %d %d %d\n",
                     ARENA_SIZE, ARENA_HALF, ARENA_XE_START);
        } else {
            snprintf(resp, sizeof(resp), "ERR no arena\n");
        }
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "crossentropy ", 13) == 0) {
        // crossentropy <kidx> <logits_off> <targets_off> <losses_off> <grad_off> <n_pos> <vocab> <inv_n>
        int kidx;
        uint32_t loff, toff, ooff, goff, npos, vocab;
        float inv_n;
        sscanf(cmd + 13, "%d %u %u %u %u %u %u %f",
               &kidx, &loff, &toff, &ooff, &goff, &npos, &vocab, &inv_n);
        int ret = dispatch_cross_entropy(kidx, loff, toff, ooff, goff, npos, vocab, inv_n);
        if (ret == 0) {
            write(fd, "OK\n", 3);
        } else {
            snprintf(resp, sizeof(resp), "ERR dispatch %d\n", ret);
            write(fd, resp, strlen(resp));
        }

    } else if (strncmp(cmd, "sync", 4) == 0) {
        xe_sync();
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "quit", 4) == 0) {
        write(fd, "OK\n", 3);
        g_running = 0;

    } else {
        snprintf(resp, sizeof(resp), "ERR unknown: %s\n", cmd);
        write(fd, resp, strlen(resp));
    }
}

int main(int argc, char** argv) {
    const char* sock_path = argc > 1 ? argv[1] : "/tmp/mongoose-xe.sock";

    signal(SIGINT, sighandler);
    signal(SIGTERM, sighandler);

    // Init Level Zero
    int ret = init_level_zero();
    if (ret != 0) {
        fprintf(stderr, "[xe-daemon] Level Zero init failed: %d\n", ret);
        return 1;
    }
    fprintf(stderr, "[xe-daemon] %s (%lu MB shared)\n",
            g_device_name, (unsigned long)g_mem_size / 1024 / 1024);

    // Init shared arena from inherited memfd (argv[2] = fd number)
    if (argc > 2) {
        int arena_fd = atoi(argv[2]);
        if (arena_fd > 0 && init_arena_from_fd(arena_fd) != 0) {
            fprintf(stderr, "[xe-daemon] WARNING: arena init failed, cross-entropy offload unavailable\n");
        }
    }

    // Create Unix socket
    unlink(sock_path);
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 1);

    // Signal parent we're ready
    printf("READY %s %lu\n", g_device_name, (unsigned long)g_mem_size);
    fflush(stdout);

    fprintf(stderr, "[xe-daemon] listening on %s\n", sock_path);

    while (g_running) {
        int client = accept(srv, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }

        // Read commands line by line
        char buf[4096];
        int pos = 0;
        while (g_running) {
            int n = read(client, buf + pos, sizeof(buf) - pos - 1);
            if (n <= 0) break;
            pos += n;
            buf[pos] = 0;

            // Process complete lines
            char* line;
            while ((line = strchr(buf, '\n')) != NULL) {
                *line = 0;
                handle_cmd(client, buf);
                int remaining = pos - (line - buf + 1);
                memmove(buf, line + 1, remaining);
                pos = remaining;
                buf[pos] = 0;
            }
        }
        close(client);
    }

    // Cleanup
    for (int i = 0; i < g_alloc_count; i++)
        zeMemFree(g_context, g_allocs[i]);
    for (int i = 0; i < g_kernel_count; i++) {
        zeKernelDestroy(g_kernels[i]);
        zeModuleDestroy(g_modules[i]);
    }
    if (g_cmdlist) zeCommandListDestroy(g_cmdlist);
    if (g_context) zeContextDestroy(g_context);

    // Cleanup arena (munmap only — Go owns the memfd lifetime)
    if (g_arena_base && g_arena_base != MAP_FAILED) {
        munmap(g_arena_base, ARENA_SIZE);
    }

    unlink(sock_path);
    fprintf(stderr, "[xe-daemon] shutdown\n");
    return 0;
}
