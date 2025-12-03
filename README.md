# ZePU: The Zero-overhead Processing Unit

**Democratizing High-Performance Computing for the AI Era.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)](https://github.com/Vivekyemky/zepu)

---

## 🚀 What is ZePU?

**ZePU (Zero-overhead Processing Unit)** is a next-generation, lightweight virtualization engine designed to transform fragmented computing resources into a unified, high-performance AI cluster.

Unlike traditional virtual machines (VMs) or heavy container orchestrators (Kubernetes), ZePU is a **micro-hypervisor** that sits closer to the metal. It abstracts away the physical hardware, allowing you to write code once and execute it across a "Hive" of distributed devices—whether they are powerful GPU servers, idle office laptops, or edge IoT devices.

It provides a Pythonic API familiar to users of **PyTorch** and **TensorFlow**, but under the hood, it manages a distributed actor model with work-stealing scheduling and transparent hardware acceleration.

---

## 🌍 The Challenge: Why ZePU?

In the current landscape of AI and High-Performance Computing (HPC), developers face three critical bottlenecks:

1.  **Resource Fragmentation:** Compute power is everywhere but disconnected. You might have a powerful workstation, a cloud instance, and edge devices, but utilizing them together requires complex networking code (MPI, RPC).
2.  **The "Tax" of Virtualization:** Docker and K8s introduce significant overhead. For latency-sensitive AI inference, every millisecond counts.
3.  **Vendor Lock-in:** AI frameworks are often tightly coupled to specific hardware (e.g., CUDA for NVIDIA). Migrating to AMD or CPU-only clusters often requires code rewrites.

### The ZePU Solution
ZePU solves these by creating a **Virtual Compute Mesh**. It treats every connected device as a "vCPU core" in a single global processor.
*   **Zero Overhead:** Written in pure C with direct memory mapping.
*   **Self-Healing:** If a node disconnects, tasks are automatically re-queued to healthy nodes.
*   **Smart Offloading:** The engine intelligently detects matrix operations and routes them to available GPUs, while keeping control flow on low-power CPUs.

---

## 💡 Use Cases

### 1. Edge AI Swarms
Connect 50 Raspberry Pis or Jetson Nanos to form a local training cluster for privacy-preserving federated learning. ZePU handles the synchronization.

### 2. Cost-Effective Model Training
Utilize "Spot Instances" or idle office computers at night. ZePU's fault tolerance means if a spot instance is reclaimed, your training job doesn't crash—it just slows down slightly.

### 3. Hybrid Cloud/On-Prem Bursting
Run your baseline workload on local hardware. When demand spikes, spin up cloud nodes that instantly join the ZePU Hive to process the extra load.

---

## 🛠️ Architecture

*   **ZePU Engine (`zepu_engine.dll` / `.so`)**: The core C runtime. Extremely small footprint (<500KB).
*   **Hive Protocol**: A custom TCP-based protocol for ultra-low latency task distribution.
*   **Smart Dispatcher**: Analyzes instruction streams in real-time.
    *   *Scalar Math* → Executed locally on vCPU.
    *   *Matrix Math* → Offloaded to `gpu_server` (CUDA/ROCm).

---

## ⚡ Quick Start

### Prerequisites
*   Python 3.8+
*   GCC (for building the core engine)

### 1. Installation
ZePU is designed to be installed as a standard Python package. This command will automatically compile the C engine for your OS.

```bash
pip install .
```

#### Manual Build (Optional)
If you need to manually rebuild the core engine (e.g., for development or cross-platform support):

**Windows (PowerShell):**
```powershell
python build_extension.py
```
*Generates `zepu/zepu_engine.dll` and `zepu/zepu_worker.exe`*

**Linux / WSL:**
```bash
python3 build_extension.py
```
*Generates `zepu/libzepu_engine.so` and `zepu/zepu_worker`*

### 2. Run a Local Cluster
Spin up a virtual cluster on your machine to test the API.

```python
from zepu import Cluster, Op

# Create a cluster with 8 virtual cores
cluster = Cluster(num_vcpus=8)

# Define a simple program (e.g., Vector Addition)
program = [
    (Op.MOV, 0, 0, 100),   # Load data
    (Op.ADD, 0, 1, 0),     # Compute
    (Op.HALT, 0, 0, 0)
]

# Load and Run
cluster.load_program(0, program)
cluster.run(cycles=1000)

print(cluster.telemetry)
```

### 3. Run the Distributed Hive (Docker)
Simulate a 10-node distributed cluster using Docker Compose.

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📦 Project Structure

| Directory | Description |
| :--- | :--- |
| **`src/`** | **The Core.** Pure C implementation of the vCPU engine and worker nodes. |
| **`zepu/`** | **The SDK.** Python package containing the `Cluster` API and distributed client. |
| **`examples/`** | **Demos.** Scripts showing PyTorch-style usage and distributed networking. |
| **`docker/`** | **Deployment.** Dockerfiles for containerized worker nodes. |

---

## 🗺️ Roadmap

*   [x] **v1.0:** Core Engine, Python Wrapper, Basic Distributed Mesh.
*   [ ] **v1.1:** WebAssembly (WASM) compilation for browser-based compute nodes.
*   [ ] **v1.2:** Auto-discovery of nodes via mDNS/Bonjour.
*   [ ] **v2.0:** JIT Compilation for Python functions directly to ZePU bytecode.

---

**Author:** Vivek Yemky
**License:** MIT
