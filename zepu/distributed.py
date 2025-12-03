"""
ZePU Distributed Cluster Client (Hive Mesh)
Author: Vivek Yemky
License: MIT
"""
import socket
import struct
import time
from .wrapper import Op

# Protocol Constants (Must match vcpu_net_proto.h)
CMD_PING        = 0
CMD_SPAWN_VCPU  = 1
CMD_LOAD_PROG   = 2
CMD_RUN_CLUSTER = 3
CMD_READ_MEM    = 4
CMD_WRITE_MEM   = 5
CMD_GET_STATS   = 6

class DistributedCluster:
    def __init__(self, nodes):
        """
        Initialize a Distributed vCPU Cluster.
        
        Args:
            nodes (list): List of strings "IP:PORT" (e.g., ["192.168.1.10:6000", "localhost:6000"])
        """
        self.nodes = []
        self.vcpu_map = [] # Maps vcpu_id -> (node_index, local_vcpu_id)
        
        print(f"[Hive] Connecting to {len(nodes)} nodes...")
        
        for node_str in nodes:
            ip, port = node_str.split(':')
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((ip, int(port)))
                self.nodes.append({
                    'sock': s,
                    'addr': node_str,
                    'vcpu_count': 0
                })
                print(f"  [+] Connected to {node_str}")
            except Exception as e:
                print(f"  [-] Failed to connect to {node_str}: {e}")

        if not self.nodes:
            raise ConnectionError("Could not connect to any worker nodes.")

    def spawn(self, total_vcpus, memory_size_kb=1024):
        """
        Distribute vCPUs across available nodes.
        """
        count_per_node = total_vcpus // len(self.nodes)
        remainder = total_vcpus % len(self.nodes)
        
        global_id = 0
        
        for i, node in enumerate(self.nodes):
            count = count_per_node + (1 if i < remainder else 0)
            if count == 0: continue
            
            # Send SPAWN command
            # Payload: { uint32_t count, uint32_t mem_size_kb }
            payload = struct.pack('II', count, memory_size_kb)
            self._send_cmd(node['sock'], CMD_SPAWN_VCPU, payload)
            self._recv_resp(node['sock']) # Wait for ACK
            
            # Update Mapping
            for local_id in range(count):
                self.vcpu_map.append((i, local_id))
                global_id += 1
            
            node['vcpu_count'] += count
            print(f"  -> Spawned {count} vCPUs on {node['addr']}")

    def load_program(self, vcpu_id, instructions):
        """
        Load program into a specific vCPU (transparently routed).
        """
        if vcpu_id >= len(self.vcpu_map):
            raise IndexError("vCPU ID out of range")
            
        node_idx, local_id = self.vcpu_map[vcpu_id]
        node = self.nodes[node_idx]
        
        # Payload: { uint32_t vcpu_id, uint32_t num_instrs } + Instructions
        header_payload = struct.pack('II', local_id, len(instructions))
        
        instr_bytes = b''
        for instr in instructions:
            # Opcode(1), R1(1), R2(1), Imm(4) -> Total 8 bytes (padded)
            # C struct: uint8, uint8, uint8, uint32 -> 7 bytes? 
            # Wait, C struct alignment usually pads. 
            # Let's check vcpu.h: 
            # struct { uint8_t opcode; uint8_t reg1; uint8_t reg2; uint32_t immediate; }
            # GCC/MSVC usually pads this to 8 bytes (1+1+1 + padding(1) + 4).
            # Let's assume packed 'BBBI' = 1+1+1+4 = 7 bytes.
            # BUT, we need to match the C struct layout exactly.
            # Safest is to send 8 bytes: B B B x I
            instr_bytes += struct.pack('BBBxI', instr[0], instr[1], instr[2], instr[3])

        self._send_cmd(node['sock'], CMD_LOAD_PROG, header_payload + instr_bytes)
        self._recv_resp(node['sock'])

    def run_all(self, cycles=100):
        """
        Broadcast RUN command to all nodes.
        """
        payload = struct.pack('Q', cycles) # uint64_t
        for node in self.nodes:
            self._send_cmd(node['sock'], CMD_RUN_CLUSTER, payload)
            self._recv_resp(node['sock'])

    def read_memory(self, vcpu_id, offset, size):
        if vcpu_id >= len(self.vcpu_map):
            raise IndexError("vCPU ID out of range")
            
        node_idx, local_id = self.vcpu_map[vcpu_id]
        node = self.nodes[node_idx]
        
        # Payload: { uint32_t vcpu_id, uint64_t offset, uint64_t size }
        payload = struct.pack('IQQ', local_id, offset, size)
        self._send_cmd(node['sock'], CMD_READ_MEM, payload)
        
        resp_payload = self._recv_resp(node['sock'])
        return resp_payload

    def get_telemetry(self):
        total_instr = 0
        total_gpu = 0
        
        for node in self.nodes:
            self._send_cmd(node['sock'], CMD_GET_STATS, b'')
            data = self._recv_resp(node['sock'])
            if data:
                instr, gpu = struct.unpack('QQ', data)
                total_instr += instr
                total_gpu += gpu
                
        return {"instructions": total_instr, "gpu_offloads": total_gpu}

    def _send_cmd(self, sock, cmd, payload):
        # Header: Magic(I), Cmd(I), ReqId(I), Len(I)
        magic = 0x56435055
        req_id = 0
        header = struct.pack('IIII', magic, cmd, req_id, len(payload))
        sock.sendall(header + payload)

    def _recv_resp(self, sock):
        # Header: Status(I), ReqId(I), Len(I)
        header_data = self._recv_all(sock, 12)
        status, req_id, length = struct.unpack('III', header_data)
        
        if status != 200:
            raise RuntimeError(f"Remote Error: {status}")
            
        if length > 0:
            return self._recv_all(sock, length)
        return None

    def _recv_all(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket closed")
            data += packet
        return data
