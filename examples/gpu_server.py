import socket
import struct
import torch
import time
import sys

HOST = '127.0.0.1'
PORT = 5555

def start_server():
    print(f"[ZePU GPU Server] Initializing PyTorch/CUDA...")
    if torch.cuda.is_available():
        device = "cuda"
        # Warmup
        torch.matmul(torch.randn(10,10, device=device), torch.randn(10,10, device=device))
    else:
        device = "cpu"
        print("[ZePU GPU Server] Warning: CUDA not available. Using CPU.")

    print(f"[ZePU GPU Server] Listening on {HOST}:{PORT}...")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        
        while True:
            conn, addr = s.accept()
            with conn:
                # print(f"[GPU Server] Connected by {addr}")
                while True:
                    data = conn.recv(4)
                    if not data:
                        break
                    
                    # Unpack size (int32)
                    size = struct.unpack('i', data)[0]
                    
                    # print(f"[GPU Server] Request: Matrix Mul {size}x{size}")
                    
                    start = time.time()
                    
                    # Perform Computation
                    a = torch.randn(size, size, device=device)
                    b = torch.randn(size, size, device=device)
                    c = torch.matmul(a, b)
                    if device == "cuda":
                        torch.cuda.synchronize()
                        
                    duration = time.time() - start
                    # print(f"[GPU Server] Done in {duration:.4f}s")
                    
                    # Send response (1 byte ack)
                    conn.sendall(b'\x01')

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n[GPU Server] Stopping...")
