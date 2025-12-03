import time
import sys
import os
from zepu.distributed import DistributedCluster
from zepu.wrapper import Op

def main():
    print("=== Docker Hive Test ===")
    
    # List of Docker Service Names
    # Docker Compose makes these resolvable by name
    nodes = [f"node{i}:6000" for i in range(1, 11)]
    
    print(f"[Client] Connecting to {len(nodes)} Docker Nodes...")
    
    # Retry logic because containers might take a second to start
    cluster = None
    for attempt in range(5):
        try:
            cluster = DistributedCluster(nodes)
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
            
    if not cluster:
        print("Failed to connect to cluster.")
        sys.exit(1)

    print("\n[Client] Spawning 10,000 vCPUs (1,000 per node)...")
    cluster.spawn(10000, memory_size_kb=64)
    
    print("[Client] Loading Task on vCPU #5000 (Node 5)...")
    # vCPU 5000 should be on Node 5 (indices 4000-4999 on Node 5? No, 0-999 Node1, 1000-1999 Node2... 4000-4999 Node5. 5000 is Node 6)
    # Let's just pick one.
    target_vcpu = 5000
    
    program = [
        (Op.MOV, 0, 0, 123),
        (Op.MOV, 1, 0, 456),
        (Op.ADD, 0, 1, 0),      # R0 = 579
        (Op.MOV, 2, 0, 64),     # R2 = Addr 64
        (Op.STORE_INDIRECT, 0, 2, 0),
        (Op.HALT, 0, 0, 0)
    ]
    cluster.load_program(target_vcpu, program)
    
    print("[Client] Running Cluster...")
    start = time.time()
    cluster.run_all(cycles=100)
    end = time.time()
    print(f"Execution Time: {end - start:.4f}s")
    
    print(f"[Client] Verifying Result from vCPU #{target_vcpu}...")
    data = cluster.read_memory(target_vcpu, offset=64, size=8)
    import struct
    result = struct.unpack('Q', data)[0]
    print(f"Result: {result}")
    
    if result == 579:
        print("SUCCESS: 10-Node Cluster Verified!")
    else:
        print("FAILURE: Result mismatch.")

    stats = cluster.get_telemetry()
    print(f"Cluster Telemetry: {stats}")

if __name__ == "__main__":
    main()
