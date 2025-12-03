"""
ZePU Engine Python Wrapper
Author: Vivek Yemky
License: MIT
"""
import ctypes
import os
import sys

# Define Structures matching vcpu.h

class VCPU_CONTEXT(ctypes.Structure):
    pass

class VCPU_CLUSTER(ctypes.Structure):
    pass

# Forward declarations
VCPU_CONTEXT._fields_ = [
    ("regs", ctypes.c_uint64 * 16),
    ("ip", ctypes.c_uint64),
    ("sp", ctypes.c_uint64),
    ("bp", ctypes.c_uint64),
    ("flags", ctypes.c_uint64),
    ("memory", ctypes.POINTER(ctypes.c_uint8)),
    ("mem_size", ctypes.c_size_t),
    ("running", ctypes.c_bool),
    ("interrupt_enabled", ctypes.c_bool),
    ("id", ctypes.c_uint32),
    ("cycle_count", ctypes.c_uint64),
    ("cluster", ctypes.POINTER(VCPU_CLUSTER))
]

VCPU_CLUSTER._fields_ = [
    ("vcpus", ctypes.POINTER(ctypes.POINTER(VCPU_CONTEXT))),
    ("vcpu_count", ctypes.c_uint32),
    # Telemetry moved here
    ("total_instructions", ctypes.c_uint64),
    ("gpu_offloads", ctypes.c_uint64),
    
    ("vcpu_capacity", ctypes.c_uint32),
    ("shared_memory", ctypes.POINTER(ctypes.c_uint8)),
    ("shared_mem_size", ctypes.c_size_t),
    ("threads", ctypes.POINTER(ctypes.c_void_p)), 
    ("thread_count", ctypes.c_uint32),
    ("parallel_execution", ctypes.c_bool),
    ("lock", ctypes.c_byte * 64), 
    ("barrier", ctypes.c_byte * 64), 
    ("global_task_index", ctypes.c_uint32)
]

class VCPU_INSTRUCTION(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint8),
        ("reg1", ctypes.c_uint8),
        ("reg2", ctypes.c_uint8),
        ("immediate", ctypes.c_uint32)
    ]

# Load Library
def load_library():
    lib_name = "zepu_engine.dll" if os.name == 'nt' else "libzepu_engine.so"
    # Try explicit path relative to this file
    lib_path = os.path.join(os.path.dirname(__file__), lib_name)
    
    if not os.path.exists(lib_path):
        # Fallback to current working directory
        lib_path = os.path.abspath(lib_name)
    
    try:
        # On Windows, we might need to add the directory to the DLL search path
        if os.name == 'nt' and sys.version_info >= (3, 8):
            os.add_dll_directory(os.path.dirname(lib_path))
        return ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"Error loading library {lib_path}: {e}")
        return None

_lib = load_library()

# Define Function Prototypes
if _lib:
    _lib.vcpu_cluster_create.argtypes = [ctypes.c_uint32, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32]
    _lib.vcpu_cluster_create.restype = ctypes.POINTER(VCPU_CLUSTER)

    _lib.vcpu_cluster_destroy.argtypes = [ctypes.POINTER(VCPU_CLUSTER)]
    _lib.vcpu_cluster_destroy.restype = None

    _lib.vcpu_cluster_execute_parallel.argtypes = [ctypes.POINTER(VCPU_CLUSTER), ctypes.c_uint64]
    _lib.vcpu_cluster_execute_parallel.restype = None

    _lib.vcpu_load_program.argtypes = [ctypes.POINTER(VCPU_CONTEXT), ctypes.POINTER(VCPU_INSTRUCTION), ctypes.c_size_t, ctypes.c_uint64]
    _lib.vcpu_load_program.restype = None

    _lib.vcpu_create_instruction.argtypes = [ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint32]
    _lib.vcpu_create_instruction.restype = VCPU_INSTRUCTION

# Pythonic Wrapper Class
class Cluster:
    def __init__(self, num_vcpus=4, thread_count=4):
        if not _lib:
            raise RuntimeError("ZePU Engine library not found. Run build_extension.py first.")
        # Increase memory to 256MB per vCPU for AI workloads
        self.obj = _lib.vcpu_cluster_create(num_vcpus, 256*1024*1024, 256*1024*1024, thread_count)
        if not self.obj:
            raise MemoryError("Failed to create ZePU Cluster")

    def __del__(self):
        if _lib and hasattr(self, 'obj') and self.obj:
            _lib.vcpu_cluster_destroy(self.obj)

    def load_program(self, vcpu_id, instructions):
        if vcpu_id >= self.obj.contents.vcpu_count:
            raise IndexError("vCPU ID out of range")
        
        # Convert list of tuples/objects to C array
        arr_type = VCPU_INSTRUCTION * len(instructions)
        c_arr = arr_type()
        
        for i, instr in enumerate(instructions):
            if isinstance(instr, dict):
                c_arr[i] = _lib.vcpu_create_instruction(
                    instr.get('op', 0), 
                    instr.get('r1', 0), 
                    instr.get('r2', 0), 
                    instr.get('imm', 0)
                )
            elif isinstance(instr, (list, tuple)):
                c_arr[i] = _lib.vcpu_create_instruction(*instr)
        
        vcpu_ptr = self.obj.contents.vcpus[vcpu_id]
        _lib.vcpu_load_program(vcpu_ptr, c_arr, len(instructions), 0)

    def run(self, cycles=100):
        _lib.vcpu_cluster_execute_parallel(self.obj, cycles)

    @property
    def telemetry(self):
        return {
            "instructions": self.obj.contents.total_instructions,
            "gpu_offloads": self.obj.contents.gpu_offloads
        }

# Opcodes
class Op:
    NOP = 0
    ADD = 1
    SUB = 2
    MUL = 3
    MOV = 5
    LOAD = 6
    STORE = 7
    JMP = 8
    CMP = 9
    JZ = 10
    JNZ = 11
    INC = 15
    DEC = 16
    STORE_INDIRECT = 27
    LOAD_INDIRECT = 28
    MATMUL = 29
    VADD = 30
    HALT = 14
