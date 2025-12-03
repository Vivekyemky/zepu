import time
import random
import sys
import os

# Add project root to path to allow importing zepu package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from zepu import Cluster, Op

def benchmark_factorial(cluster, vcpu_id, n):
    """
    Calculates Factorial(n) iteratively.
    Tests: Arithmetic, Branching, Loop Overhead.
    """
    # Reg0: n (Input)
    # Reg1: Result (Accumulator)
    # Reg2: Counter (Loop Var)
    # Reg3: Temp/Constant 1
    
    program = [
        (Op.MOV, 0, 0, n),      # R0 = n
        (Op.MOV, 1, 0, 1),      # R1 = 1 (Result)
        (Op.MOV, 3, 0, 1),      # R3 = 1 (Constant)
        
        # Loop Label (Index 3)
        (Op.CMP, 0, 0, 1),      # Compare n with 1
        (Op.JZ, 0, 0, 8),       # If n == 1, Jump to End (Index 8)
        
        (Op.MUL, 1, 0, 0),      # R1 = R1 * R0
        (Op.SUB, 0, 0, 3),      # R0 = R0 - 1
        (Op.JMP, 0, 0, 3),      # Jump back to Loop (Index 3)
        
        # End Label (Index 8)
        (Op.HALT, 0, 0, 0)
    ]
    
    cluster.load_program(vcpu_id, program)
    
def benchmark_vector_add(cluster, vcpu_id, size):
    """
    Calculates C[i] = A[i] + B[i] for i in 0..size.
    Tests: Memory Bandwidth, Indirect Addressing.
    """
    base_a = 1000
    base_b = 2000
    base_c = 3000
    
    # Reg0: Ptr A
    # Reg1: Ptr B
    # Reg2: Ptr C
    # Reg3: Count
    # Reg4: Temp Val A
    # Reg5: Temp Val B
    # Reg6: Constant 8 (sizeof uint64)
    
    program = [
        (Op.MOV, 0, 0, base_a), # R0 = Base A
        (Op.MOV, 1, 0, base_b), # R1 = Base B
        (Op.MOV, 2, 0, base_c), # R2 = Base C
        (Op.MOV, 3, 0, size),   # R3 = Count
        (Op.MOV, 6, 0, 8),      # R6 = 8 (Stride)
        
        # Loop Label (Index 5)
        (Op.CMP, 3, 0, 0),      # Compare Count with 0
        (Op.JZ, 0, 0, 15),      # If Count == 0, Jump to End
        
        (Op.LOAD_INDIRECT, 4, 0, 0), # R4 = *R0
        (Op.LOAD_INDIRECT, 5, 0, 1), # R5 = *R1
        (Op.ADD, 4, 0, 5),           # R4 = R4 + R5
        (Op.STORE_INDIRECT, 4, 0, 2),# *R2 = R4
        
        # Increment Pointers
        (Op.ADD, 0, 0, 6),      # R0 += 8
        (Op.ADD, 1, 0, 6),      # R1 += 8
        (Op.ADD, 2, 0, 6),      # R2 += 8
        
        # Decrement Count
        (Op.DEC, 3, 0, 0),      # R3--
        (Op.JMP, 0, 0, 5),      # Jump to Loop
        
        # End
        (Op.HALT, 0, 0, 0)
    ]
    cluster.load_program(vcpu_id, program)

def benchmark_gpu_matmul(cluster, vcpu_id, size):
    """
    Offloads Matrix Multiplication to GPU.
    Tests: Hardware Acceleration, Latency.
    """
    program = [
        (Op.MOV, 0, 0, 1000),   # Ptr A
        (Op.MOV, 1, 0, 2000),   # Ptr B
        (Op.MOV, 3, 0, size),   # Size (Triggers GPU if >= 2048)
        (Op.MATMUL, 0, 1, 3000),# C = A * B
        (Op.HALT, 0, 0, 0)
    ]
    cluster.load_program(vcpu_id, program)

def main():
    print("=== ZePU Enterprise Performance Suite ===")
    print("Initializing Cluster with 8 vCPUs / 8 Threads...")
    cluster = Cluster(num_vcpus=8, thread_count=8)
    
    # 1. CPU Stress Test (Factorial)
    print("\n[Test 1] CPU Compute Stress (Factorial Calculation)")
    print("    Running Factorial(20) on 4 vCPUs in parallel...")
    for i in range(4):
        benchmark_factorial(cluster, i, 20)
    
    start = time.time()
    cluster.run(cycles=1000) # Should be enough for Fac(20)
    end = time.time()
    
    stats = cluster.telemetry
    print(f"    Time: {end-start:.6f}s")
    print(f"    Total Instructions: {stats['instructions']}")
    print(f"    IPS: {stats['instructions'] / (end-start):.2f}")

    # 2. Memory Bandwidth Test (Vector Add)
    print("\n[Test 2] Memory Bandwidth (Vector Addition)")
    print("    Running VectorAdd(1000 elements) on vCPU 4...")
    benchmark_vector_add(cluster, 4, 1000)
    
    start = time.time()
    cluster.run(cycles=10000) # 1000 elements * ~10 instrs = 10k cycles
    end = time.time()
    
    stats_new = cluster.telemetry
    instrs = stats_new['instructions'] - stats['instructions']
    print(f"    Time: {end-start:.6f}s")
    print(f"    Instructions Executed: {instrs}")
    print(f"    Throughput: {instrs / (end-start):.2f} ops/sec")

    # 3. Hardware Acceleration Test
    print("\n[Test 3] Hardware Acceleration (Matrix Multiplication)")
    print("    Offloading 2048x2048 Matrix Mul to GPU Server...")
    benchmark_gpu_matmul(cluster, 5, 2048)
    
    start = time.time()
    cluster.run(cycles=100) # Only needs a few cycles to dispatch
    end = time.time()
    
    stats_final = cluster.telemetry
    offloads = stats_final['gpu_offloads'] - stats_new['gpu_offloads']
    
    print(f"    Time: {end-start:.6f}s")
    print(f"    GPU Offloads: {offloads}")
    if offloads > 0:
        print("    STATUS: \033[92mHARDWARE ACCELERATION ACTIVE\033[0m")
    else:
        print("    STATUS: \033[93mRUNNING ON CPU (Server not detected)\033[0m")

    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()