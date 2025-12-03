import os
import subprocess
import sys
import argparse

def build(target=None):
    print("Building vCPU Engine Shared Library...")
    
    src = os.path.join("src", "vcpu_core.c")
    
    # Determine Target OS
    is_windows_host = os.name == 'nt'
    
    # Default to Host OS unless overridden
    if target == 'linux':
        target_is_windows = False
    elif target == 'windows':
        target_is_windows = True
    else:
        target_is_windows = is_windows_host

    # Set Output Name
    if target_is_windows:
        out = os.path.join("zepu", "zepu_engine.dll")
    else:
        out = os.path.join("zepu", "libzepu_engine.so")

    # Build Command
    cmd = ["gcc", "-shared", "-o", out, src, "-I", "src"]
    
    # Add Flags
    if is_windows_host:
        # We are compiling ON Windows (MinGW)
        # Even if we call it .so, it will be a PE file and needs Windows libs
        cmd.extend(["-lpthread", "-lws2_32", "-O3", "-static"])
        if not target_is_windows:
            print("WARNING: Building a .so file on Windows. This will be a Windows PE binary named .so, NOT a Linux ELF binary.")
    else:
        # We are compiling ON Linux
        cmd.extend(["-lpthread", "-fPIC", "-O3"])

    print(f"Target: {'Windows' if target_is_windows else 'Linux (Simulated)'}")
    print(f"Output: {out}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print(f"Successfully built {out}")
    except subprocess.CalledProcessError as e:
        print("Build failed!")
        sys.exit(1)

    # --- Build Worker Node ---
    print("\nBuilding Worker Node Executable...")
    worker_src = os.path.join("src", "vcpu_worker.c")
    core_src = os.path.join("src", "vcpu_core.c")
    
    if is_windows_host:
        worker_out = os.path.join("zepu", "zepu_worker.exe")
        # Link against the DLL we just built? Or compile static?
        # Easier to compile static for standalone worker.
        # We need to include vcpu_core.c in the compilation or link the object.
        # Let's compile vcpu_worker.c AND vcpu_core.c together into one exe.
        worker_cmd = ["gcc", "-o", worker_out, worker_src, core_src, "-I", "src", "-lpthread", "-lws2_32", "-O3", "-static"]
    else:
        worker_out = os.path.join("zepu", "zepu_worker")
        worker_cmd = ["gcc", "-o", worker_out, worker_src, core_src, "-I", "src", "-lpthread", "-O3"]

    print(f"Running: {' '.join(worker_cmd)}")
    try:
        subprocess.check_call(worker_cmd)
        print(f"Successfully built {worker_out}")
    except subprocess.CalledProcessError as e:
        print("Worker Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build vCPU Engine')
    parser.add_argument('--target', choices=['windows', 'linux'], help='Force build target')
    args = parser.parse_args()
    
    build(args.target)