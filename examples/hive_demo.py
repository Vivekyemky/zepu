import time
import subprocess
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from zepu.distributed import DistributedCluster
from zepu.wrapper import Op

def main():
    print("=== ZePU Hive: Distributed Cluster Demo ===")
    
    # 1. Start a Local Worker Node (Background Process)
    # In a real scenario, this would be running on a different machine.
    print("[Host] Launching ZePU Worker Node on Port 6000...")
    
    # Determine executable extension
    exe_ext = ".exe" if os.name == 'nt' else ""
    # We are in zepu/hive_demo.py, so we need to go up one level to find the exe
    worker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", f"zepu_worker{exe_ext}"))
    
    print(f"[Host] Looking for worker at: {worker_path}")
    
    # We assume the worker is compiled. If not, we might fail.
    # Let's try to run it.
    try:
        worker_proc = subprocess.Popen([worker_path, "6000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1) # Give it time to bind
    except FileNotFoundError:
        print("Error: zepu_worker executable not found. Please compile it first.")
        return

    try:
        # 2. Connect to the Hive
        # We connect to localhost:6000. You could add "192.168.x.x:6000" here too.
        cluster = DistributedCluster(nodes=["127.0.0.1:6000"])
        
        # 3. Spawn 100 vCPUs
        print("\n[Host] Spawning 100 vCPUs...")
        cluster.spawn(100, memory_size_kb=64)
        
        # 4. Load a Program onto vCPU #50 (which lives on the worker)
        print("[Host] Loading Task onto vCPU #50...")
        program = [
            (Op.MOV, 0, 0, 100),
            (Op.MOV, 1, 0, 200),
            # Fix: ADD uses Reg1 and Reg2. Imm is unused.
            # We want ADD R0, R1. So Reg1=0, Reg2=1.
            (Op.ADD, 0, 1, 0),   # R0 = 300
            (Op.HALT, 0, 0, 0)
        ]
        cluster.load_program(50, program)
        
        # 5. Run the Cluster
        print("[Host] Broadcasting RUN command...")
        cluster.run_all(cycles=100)
        
        # 6. Verify Result
        # We will store the result at offset 128 (safe from code overwrite)
        print("[Host] Updating Task to Store Result at Offset 128...")
        program_store = [
            (Op.MOV, 0, 0, 100),
            (Op.MOV, 1, 0, 200),
            # Fix: ADD R0, R1
            (Op.ADD, 0, 1, 0),      # R0 = 300
            (Op.MOV, 2, 0, 128),    # R2 = Address 128
            # Fix: STORE_INDIRECT uses Reg1 (Src) and Reg2 (Addr). Imm is unused.
            # We want Store R0 to [R2]. So Reg1=0, Reg2=2.
            (Op.STORE_INDIRECT, 0, 2, 0), 
            (Op.HALT, 0, 0, 0)
        ]
        cluster.load_program(50, program_store)
        cluster.run_all(cycles=100)
        
        # 7. Read Memory from vCPU #50
        print("[Host] Reading Result from vCPU #50 (Offset 128)...")
        data = cluster.read_memory(50, offset=128, size=8)
        import struct
        result = struct.unpack('Q', data)[0]
        print(f"Result (Int): {result}")
        print(f"Result (Hex): {data.hex()}")
        
        if result == 300:
            print("SUCCESS: Distributed Execution Verified!")
        else:
            print("FAILURE: Result mismatch.")

        # 8. Telemetry
        stats = cluster.get_telemetry()
        print(f"\nTelemetry: {stats}")

    finally:
        print("\n[Host] Shutting down worker...")
        worker_proc.terminate()
        try:
            outs, errs = worker_proc.communicate(timeout=2)
            print("\n--- Worker Logs ---")
            print(outs.decode('utf-8', errors='ignore'))
            print("--- Worker Errors ---")
            print(errs.decode('utf-8', errors='ignore'))
        except:
            pass

if __name__ == "__main__":
    main()
