// mongoose Metal compute — native Swift, native Metal.
// Compiled as standalone binary. Go shells out via stdin/stdout binary protocol.
//
// Build: swiftc -O -o mongoose-metal mongoose_metal.swift -framework Metal -framework MetalPerformanceShaders
//
// Protocol (binary on stdin/stdout):
//   Request:  [1B opcode][4B M][4B K][4B N][M*K*4B matrix A][K*N*4B matrix B]
//   Response: [4B size][M*N*4B result]
//
// Opcodes:
//   0x01: MatMul (C = A @ B)
//   0x02: RMSNorm (in-place, weight follows data)
//   0x03: ReLU (in-place)
//   0x04: Benchmark
//   0xFF: Quit

import Foundation
import Metal
import MetalPerformanceShaders

// MARK: - Metal Setup

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write("mongoose-metal: no Metal device\n".data(using: .utf8)!)
    exit(1)
}

guard let commandQueue = device.makeCommandQueue() else {
    FileHandle.standardError.write("mongoose-metal: no command queue\n".data(using: .utf8)!)
    exit(1)
}

let deviceName = device.name
FileHandle.standardError.write("mongoose-metal: \(deviceName)\n".data(using: .utf8)!)

// MARK: - Binary I/O helpers

let stdin = FileHandle.standardInput
let stdout = FileHandle.standardOutput

func readExact(_ n: Int) -> Data? {
    var buf = Data()
    while buf.count < n {
        let chunk = stdin.readData(ofLength: n - buf.count)
        if chunk.isEmpty { return nil }
        buf.append(chunk)
    }
    return buf
}

func readUInt32() -> UInt32? {
    guard let d = readExact(4) else { return nil }
    return d.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
}

func readFloat32Array(_ count: Int) -> [Float]? {
    guard let d = readExact(count * 4) else { return nil }
    return d.withUnsafeBytes { ptr in
        Array(ptr.bindMemory(to: Float.self))
    }
}

func writeFloat32Array(_ arr: [Float]) {
    let size = UInt32(arr.count * 4).littleEndian
    var sizeBytes = size
    let sizeData = Data(bytes: &sizeBytes, count: 4)
    stdout.write(sizeData)
    arr.withUnsafeBytes { ptr in
        stdout.write(Data(ptr))
    }
}

// MARK: - Metal MatMul via MPS

func metalMatMul(a: [Float], b: [Float], m: Int, k: Int, n: Int) -> [Float] {
    // Create Metal buffers
    let aBuffer = device.makeBuffer(bytes: a, length: m * k * 4, options: .storageModeShared)!
    let bBuffer = device.makeBuffer(bytes: b, length: k * n * 4, options: .storageModeShared)!
    let cBuffer = device.makeBuffer(length: m * n * 4, options: .storageModeShared)!

    // MPS matrix descriptors
    let aDesc = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * 4, dataType: .float32)
    let bDesc = MPSMatrixDescriptor(rows: k, columns: n, rowBytes: n * 4, dataType: .float32)
    let cDesc = MPSMatrixDescriptor(rows: m, columns: n, rowBytes: n * 4, dataType: .float32)

    let matA = MPSMatrix(buffer: aBuffer, descriptor: aDesc)
    let matB = MPSMatrix(buffer: bBuffer, descriptor: bDesc)
    let matC = MPSMatrix(buffer: cBuffer, descriptor: cDesc)

    // Create matmul kernel
    let matmul = MPSMatrixMultiplication(device: device,
                                          transposeLeft: false,
                                          transposeRight: false,
                                          resultRows: m,
                                          resultColumns: n,
                                          interiorColumns: k,
                                          alpha: 1.0,
                                          beta: 0.0)

    // Encode and run
    let cmdBuf = commandQueue.makeCommandBuffer()!
    matmul.encode(commandBuffer: cmdBuf, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // Read result
    let cPtr = cBuffer.contents().bindMemory(to: Float.self, capacity: m * n)
    return Array(UnsafeBufferPointer(start: cPtr, count: m * n))
}

// MARK: - Benchmark

func benchmark() -> Double {
    let dim = 1024
    let a = [Float](repeating: 0.001, count: dim * dim)
    let b = [Float](repeating: 0.001, count: dim * dim)

    // Warmup
    _ = metalMatMul(a: a, b: b, m: dim, k: dim, n: dim)

    // Benchmark
    let iters = 100
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        _ = metalMatMul(a: a, b: b, m: dim, k: dim, n: dim)
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    let flops = Double(2 * dim * dim * dim * iters)
    return flops / elapsed / 1e9 // GFLOPS
}

// MARK: - Main Loop

while true {
    guard let opData = readExact(1) else { break }
    let op = opData[0]

    switch op {
    case 0x01: // MatMul
        guard let m = readUInt32(), let k = readUInt32(), let n = readUInt32() else { break }
        guard let a = readFloat32Array(Int(m * k)), let b = readFloat32Array(Int(k * n)) else { break }
        let c = metalMatMul(a: a, b: b, m: Int(m), k: Int(k), n: Int(n))
        writeFloat32Array(c)

    case 0x04: // Benchmark
        let gflops = benchmark()
        var result = Float(gflops)
        writeFloat32Array([result])
        FileHandle.standardError.write("mongoose-metal: \(String(format: "%.1f", gflops)) GFLOPS on \(deviceName)\n".data(using: .utf8)!)

    case 0xFF: // Quit
        exit(0)

    default:
        FileHandle.standardError.write("mongoose-metal: unknown opcode \(op)\n".data(using: .utf8)!)
    }
}
