#include "../gpu_utils.h"

// ReLU activation kernel
__global__ void relu_forward_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU backward kernel
__global__ void relu_backward_kernel(float* grad_output, float* grad_input, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// Add bias kernel
__global__ void add_bias_kernel(float* output, float* bias, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int channel = (idx / (height * width)) % channels;
        output[idx] += bias[channel];
    }
}

// Element-wise operations
__global__ void add_kernel(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void scale_kernel(float* data, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] *= scale;
    }
}

// MSE Loss kernel
__global__ void mse_loss_kernel(float* output, float* target, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float diff = output[idx] - target[idx];
        atomicAdd(loss, diff * diff);
    }
}

// MSE gradient kernel
__global__ void mse_gradient_kernel(float* output, float* target, float* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad_output[idx] = 2.0f * (output[idx] - target[idx]) / size;
    }
}

// C++ wrapper functions
extern "C" {
    void relu_forward_gpu(float* input, float* output, int size) {
        dim3 grid = get_grid_size(size, gpu_config::BLOCK_SIZE_1D);
        dim3 block(gpu_config::BLOCK_SIZE_1D);
        
        relu_forward_kernel<<<grid, block>>>(input, output, size);
        CUDA_CHECK_KERNEL();
    }
    
    void relu_backward_gpu(float* grad_output, float* grad_input, float* input, int size) {
        dim3 grid = get_grid_size(size, gpu_config::BLOCK_SIZE_1D);
        dim3 block(gpu_config::BLOCK_SIZE_1D);
        
        relu_backward_kernel<<<grid, block>>>(grad_output, grad_input, input, size);
        CUDA_CHECK_KERNEL();
    }
    
    void add_bias_gpu(float* output, float* bias, int batch_size, int channels, int height, int width) {
        int total_size = batch_size * channels * height * width;
        dim3 grid = get_grid_size(total_size, gpu_config::BLOCK_SIZE_1D);
        dim3 block(gpu_config::BLOCK_SIZE_1D);
        
        add_bias_kernel<<<grid, block>>>(output, bias, batch_size, channels, height, width);
        CUDA_CHECK_KERNEL();
    }
    
    float mse_loss_gpu(float* output, float* target, int size) {
        float* d_loss;
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        
        dim3 grid = get_grid_size(size, gpu_config::BLOCK_SIZE_1D);
        dim3 block(gpu_config::BLOCK_SIZE_1D);
        
        mse_loss_kernel<<<grid, block>>>(output, target, d_loss, size);
        CUDA_CHECK_KERNEL();
        
        float h_loss;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_loss));
        
        return h_loss / size;
    }
    
    void mse_gradient_gpu(float* output, float* target, float* grad_output, int size) {
        dim3 grid = get_grid_size(size, gpu_config::BLOCK_SIZE_1D);
        dim3 block(gpu_config::BLOCK_SIZE_1D);
        
        mse_gradient_kernel<<<grid, block>>>(output, target, grad_output, size);
        CUDA_CHECK_KERNEL();
    }
}