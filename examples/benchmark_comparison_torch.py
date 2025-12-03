import time
import torch
import sys
import os

# Add current directory to path to import zepu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from zepu import Cluster, Op

def benchmark_torch_factorial(n):
    """
    Simulate a heavy loop.
    """
    start = time.time()
    result = 1
    # Increase n to 1,000,000 for measurable time
    for i in range(1, 1000000):
        result = (result * i) % 100000 # Prevent massive integer expansion
    end = time.time()
    return end - start

def benchmark_torch_vector_add(size):
    """
    PyTorch Vector Addition (Highly Optimized C++/AVX).
    """
    a = torch.ones(size)
    b = torch.ones(size)
    
    start = time.time()
    c = a + b
    end = time.time()
    return end - start

def benchmark_torch_matmul(size):
    """
    PyTorch Matrix Multiplication (BLAS/CUDA).
    """
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    start = time.time()
    c = torch.matmul(a, b)
    end = time.time()
    return end - start

def run_vcpu_benchmarks():
    print("\n--- ZePU Benchmarks ---")
    cluster = Cluster(num_vcpus=1, thread_count=1)
    
    # 1. Logic Loop (1,000,000 iterations)
    # We'll just decrement a counter from 1,000,000 to 0
    # Optimization: DEC now updates flags, so we don't need CMP!
    program_loop = [
        (Op.MOV, 0, 0, 1000000), # R0 = 1,000,000
        (Op.DEC, 0, 0, 0),       # R0-- (Updates ZF)
        (Op.JNZ, 0, 0, 1 * 8),   # If != 0, Jump back to Index 1 (DEC)
        (Op.HALT, 0, 0, 0)
    ]
    cluster.load_program(0, program_loop)
    start = time.time()
    # 1M iterations * 2 instructions (DEC, JNZ) = 2M cycles
    cluster.run(cycles=3000000) 
    vcpu_loop_time = time.time() - start
    print(f"vCPU Loop(1M):        {vcpu_loop_time:.6f}s")

    # 2. Vector Add (100,000 elements)
    # 100k elements to get measurable time
    size = 100000
    program_vec = [
        (Op.MOV, 0, 0, 1000), (Op.MOV, 1, 0, 2000), (Op.MOV, 2, 0, 3000),
        (Op.MOV, 3, 0, size), (Op.MOV, 6, 0, 8),
        (Op.CMP, 3, 0, 0), (Op.JZ, 0, 0, 16 * 8), # Jump to HALT (Index 16)
        (Op.LOAD_INDIRECT, 4, 0, 0), (Op.LOAD_INDIRECT, 5, 0, 1),
        (Op.ADD, 4, 0, 5), (Op.STORE_INDIRECT, 4, 0, 2),
        (Op.ADD, 0, 0, 6), (Op.ADD, 1, 0, 6), (Op.ADD, 2, 0, 6),
        (Op.DEC, 3, 0, 0), (Op.JMP, 0, 0, 5 * 8), # Jump to CMP (Index 5)
        (Op.HALT, 0, 0, 0)
    ]
    cluster.load_program(0, program_vec)
    start = time.time()
    # 100k * ~10 instructions = 1M cycles
    cluster.run(cycles=2000000)
    vcpu_vec_time = time.time() - start
    print(f"vCPU VectorAdd(100k): {vcpu_vec_time:.6f}s")

    # 2b. Vector Add (Optimized VADD Opcode)
    # Using the new SIMD-friendly instruction
    # Increase to 10M to amortize thread creation cost and measure throughput
    size_opt = 10000000
    # Offset pointers to avoid overlap (40MB per array)
    # A: 0, B: 50MB, C: 100MB
    program_vadd = [
        (Op.MOV, 0, 0, 0), 
        (Op.MOV, 1, 0, 50000000), 
        (Op.MOV, 3, 0, size_opt), # Size 10M
        (Op.VADD, 0, 1, 100000000), 
        (Op.HALT, 0, 0, 0)
    ]
    cluster.load_program(0, program_vadd)
    start = time.time()
    cluster.run(cycles=1000) 
    vcpu_vadd_time = time.time() - start
    print(f"vCPU VADD(10M):       {vcpu_vadd_time:.6f}s (Optimized)")

    # 3. MatMul (2048x2048) - Offloaded
    program_matmul = [
        (Op.MOV, 0, 0, 1000), (Op.MOV, 1, 0, 2000), (Op.MOV, 3, 0, 2048),
        (Op.MATMUL, 0, 1, 3000), (Op.HALT, 0, 0, 0)
    ]
    cluster.load_program(0, program_matmul)
    start = time.time()
    cluster.run(cycles=100)
    vcpu_matmul_time = time.time() - start
    print(f"vCPU MatMul(2048):    {vcpu_matmul_time:.6f}s (Dispatch Time)")
    
    return vcpu_loop_time, vcpu_vec_time, vcpu_vadd_time, vcpu_matmul_time

def main():
    print("=== Comparative Benchmark: vCPU vs PyTorch/Python ===")
    
    # Run Torch/Python Benchmarks
    print("\n--- PyTorch/Python Benchmarks ---")
    py_loop_time = benchmark_torch_factorial(0) # Argument ignored, fixed to 1M inside
    print(f"Python Loop(1M):      {py_loop_time:.6f}s")
    
    # Run 10M to get measurable time, then scale down
    torch_vec_time = benchmark_torch_vector_add(10000000)
    torch_vec_time_100k = torch_vec_time / 100.0
    print(f"Torch VectorAdd(10M): {torch_vec_time:.6f}s (Est 100k: {torch_vec_time_100k:.6f}s)")
    
    torch_matmul_time = benchmark_torch_matmul(2048)
    print(f"Torch MatMul(2048):   {torch_matmul_time:.6f}s")

    # Run vCPU Benchmarks
    vcpu_loop, vcpu_vec, vcpu_vadd, vcpu_matmul = run_vcpu_benchmarks()

    # Comparison
    print("\n=== Results Analysis ===")
    print(f"1. Control Flow (1M Loops):")
    print(f"   vCPU Time: {vcpu_loop:.4f}s | Python Time: {py_loop_time:.4f}s")
    if vcpu_loop < py_loop_time:
        print(f"   -> vCPU is {py_loop_time / vcpu_loop:.2f}x FASTER than Python Loop")
    else:
        print(f"   -> vCPU is {vcpu_loop / py_loop_time:.2f}x SLOWER than Python Loop")
    
    print(f"\n2. Vector Math (10M Elements):")
    print(f"   vCPU Optimized: {vcpu_vadd:.6f}s | PyTorch: {torch_vec_time:.6f}s")
    
    if torch_vec_time > 0:
        if vcpu_vadd < torch_vec_time:
             print(f"   -> vCPU is {torch_vec_time / vcpu_vadd:.2f}x FASTER than PyTorch")
        else:
             print(f"   -> PyTorch is {vcpu_vadd / torch_vec_time:.2f}x FASTER than vCPU")

    print(f"\n3. Matrix Multiplication (2048x2048):")
    print(f"   vCPU Dispatch: {vcpu_matmul:.4f}s | PyTorch Exec: {torch_matmul_time:.4f}s")
    print(f"   -> vCPU offloads asynchronously. Real execution happens on Server.")

if __name__ == "__main__":
    main()