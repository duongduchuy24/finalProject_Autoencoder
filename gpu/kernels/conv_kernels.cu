#include "../gpu_utils.h"

// Simple convolution kernel (naive implementation)
__global__ void conv2d_forward_kernel(
    float* input, float* output, float* weights, float* bias,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_height * output_width;
    
    if (idx < total_outputs) {
        // Decode indices
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int oc = (idx / (output_width * output_height)) % out_channels;
        int b = idx / (out_channels * output_height * output_width);
        
        float sum = 0.0f;
        
        // Convolution operation
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = h * stride - padding + kh;
                    int iw = w * stride - padding + kw;
                    
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        int input_idx = b * (in_channels * input_height * input_width) +
                                       ic * (input_height * input_width) +
                                       ih * input_width + iw;
                        
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                        ic * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;
                        
                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        
        output[idx] = sum + bias[oc];
    }
}

// Max pooling kernel
__global__ void maxpool2d_forward_kernel(
    float* input, float* output, int* indices,
    int batch_size, int channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int pool_size, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_height * output_width;
    
    if (idx < total_outputs) {
        // Decode indices
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int b = idx / (channels * output_height * output_width);
        
        float max_val = -FLT_MAX;
        int max_idx = -1;
        
        // Pool operation
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int ih = h * stride + ph;
                int iw = w * stride + pw;
                
                if (ih < input_height && iw < input_width) {
                    int input_idx = b * (channels * input_height * input_width) +
                                   c * (input_height * input_width) +
                                   ih * input_width + iw;
                    
                    if (input[input_idx] > max_val) {
                        max_val = input[input_idx];
                        max_idx = input_idx;
                    }
                }
            }
        }
        
        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

// Upsampling kernel (nearest neighbor)
__global__ void upsample2d_forward_kernel(
    float* input, float* output,
    int batch_size, int channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int scale_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_height * output_width;
    
    if (idx < total_outputs) {
        // Decode indices
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int b = idx / (channels * output_height * output_width);
        
        // Map to input coordinates
        int ih = h / scale_factor;
        int iw = w / scale_factor;
        
        int input_idx = b * (channels * input_height * input_width) +
                       c * (input_height * input_width) +
                       ih * input_width + iw;
        
        output[idx] = input[input_idx];
    }
}

// C++ wrapper functions
extern "C" {
    void conv2d_forward_gpu(
        float* input, float* output, float* weights, float* bias,
        int batch_size, int in_channels, int out_channels,
        int input_height, int input_width,
        int output_height, int output_width,
        int kernel_size, int stride, int padding
    ) {
        int total_outputs = batch_size * out_channels * output_height * output_width;
        dim3 grid = get_grid_size(total_outputs, gpu_config::BLOCK_SIZE_1D);
        dim3 block(gpu_config::BLOCK_SIZE_1D);
        
        conv2d_forward_kernel<<<grid, block>>>(
            input, output, weights, bias,
            batch_size, in_channels, out_channels,
            input_height, input_width, output_height, output_width,
            kernel_size, stride, padding
        );
        CUDA_CHECK_KERNEL();
    }
    
    void maxpool2d_forward_gpu(
        float* input, float* output, int* indices,
        int batch_size, int channels,
        int input_height, int input_width,
        int output_height, int output_width,
        int pool_size, int stride
    ) {
        int total_outputs = batch_size * channels * output_height * output_width;
        dim3 grid = get_grid_size(total_outputs, gpu_config::BLOCK_SIZE_1D);
        dim3 block(gpu_config::BLOCK_SIZE_1D);
        
        maxpool2d_forward_kernel<<<grid, block>>>(
            input, output, indices,
            batch_size, channels, input_height, input_width,
            output_height, output_width, pool_size, stride
        );
        CUDA_CHECK_KERNEL();
    }
    
    void upsample2d_forward_gpu(
        float* input, float* output,
        int batch_size, int channels,
        int input_height, int input_width,
        int output_height, int output_width,
        int scale_factor
    ) {
        int total_outputs = batch_size * channels * output_height * output_width;
        dim3 grid = get_grid_size(total_outputs, gpu_config::BLOCK_SIZE_1D);
        dim3 block(gpu_config::BLOCK_SIZE_1D);
        
        upsample2d_forward_kernel<<<grid, block>>>(
            input, output, batch_size, channels,
            input_height, input_width, output_height, output_width,
            scale_factor
        );
        CUDA_CHECK_KERNEL();
    }
}