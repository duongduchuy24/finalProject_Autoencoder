#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

// GPU memory management utilities
template<typename T>
class GPUMemory {
private:
    T* d_ptr;
    size_t size_bytes;

public:
    GPUMemory() : d_ptr(nullptr), size_bytes(0) {}
    
    explicit GPUMemory(size_t count) : d_ptr(nullptr), size_bytes(count * sizeof(T)) {
        allocate(count);
    }
    
    ~GPUMemory() {
        free();
    }
    
    void allocate(size_t count) {
        free();
        size_bytes = count * sizeof(T);
        CUDA_CHECK(cudaMalloc(&d_ptr, size_bytes));
    }
    
    void free() {
        if (d_ptr) {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
            size_bytes = 0;
        }
    }
    
    void copyFromHost(const T* h_ptr, size_t count) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copyToHost(T* h_ptr, size_t count) const {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    void memset(int value = 0) {
        CUDA_CHECK(cudaMemset(d_ptr, value, size_bytes));
    }
    
    T* get() const { return d_ptr; }
    size_t bytes() const { return size_bytes; }
};

// Common GPU constants
namespace gpu_config {
    constexpr int BLOCK_SIZE_1D = 256;
    constexpr int BLOCK_SIZE_2D = 16;
    constexpr int MAX_THREADS_PER_BLOCK = 1024;
}

// Helper functions
inline dim3 get_grid_size(int n, int block_size) {
    return dim3((n + block_size - 1) / block_size);
}

inline dim3 get_grid_size_2d(int h, int w, int block_size) {
    return dim3((w + block_size - 1) / block_size, (h + block_size - 1) / block_size);
}