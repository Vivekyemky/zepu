# Stage 1: Build
FROM gcc:11 as builder

WORKDIR /build
COPY src/vcpu_worker.c .
COPY src/vcpu_core.c .
COPY src/vcpu.h .
COPY src/vcpu_net_proto.h .
# We need windows_compat.h if it's included, even if unused on Linux
COPY src/windows_compat.h . 

# Compile statically to avoid dependency hell, or just compile for this OS
RUN gcc -o zepu_worker vcpu_worker.c vcpu_core.c -lpthread -O3

# Stage 2: Runtime
FROM debian:bullseye-slim

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /build/zepu_worker /app/zepu_worker

# Expose the port
EXPOSE 6000

# Run the worker
CMD ["./zepu_worker", "6000"]
