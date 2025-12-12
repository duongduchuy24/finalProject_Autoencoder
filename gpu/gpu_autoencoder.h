#pragma once

#include "../common/types.h"
#include "gpu_utils.h"
#include <memory>

// Forward declarations for kernel functions
extern "C" {
    void conv2d_forward_gpu(
        float* input, float* output, float* weights, float* bias,
        int batch_size, int in_channels, int out_channels,
        int input_height, int input_width,
        int output_height, int output_width,
        int kernel_size, int stride, int padding
    );
    
    void maxpool2d_forward_gpu(
        float* input, float* output, int* indices,
        int batch_size, int channels,
        int input_height, int input_width,
        int output_height, int output_width,
        int pool_size, int stride
    );
    
    void upsample2d_forward_gpu(
        float* input, float* output,
        int batch_size, int channels,
        int input_height, int input_width,
        int output_height, int output_width,
        int scale_factor
    );
    
    void relu_forward_gpu(float* input, float* output, int size);
    float mse_loss_gpu(float* output, float* target, int size);
    void mse_gradient_gpu(float* output, float* target, float* grad_output, int size);
}

class GPUAutoencoder {
private:
    real_t learning_rate;
    
    // Network architecture parameters
    struct LayerConfig {
        int input_channels, output_channels, kernel_size, stride, padding;
        int input_height, input_width, output_height, output_width;
    };
    
    std::vector<LayerConfig> layer_configs;
    
    // GPU memory for weights and biases
    std::vector<GPUMemory<float>> d_weights;
    std::vector<GPUMemory<float>> d_bias;
    
    // GPU memory for activations (forward pass)
    std::vector<GPUMemory<float>> d_activations;
    
    // GPU memory for pooling indices
    std::vector<GPUMemory<int>> d_pool_indices;
    
    // Input and output buffers
    GPUMemory<float> d_input;
    GPUMemory<float> d_output;
    GPUMemory<float> d_target;
    
    // Initialize network architecture
    void init_architecture();
    void init_gpu_memory(int batch_size);
    void init_weights();
    
public:
    GPUAutoencoder(real_t lr = 0.001f);
    ~GPUAutoencoder();
    
    void forward(const Tensor4D& input, Tensor4D& output);
    void encode(const Tensor4D& input, Tensor4D& encoded);
    real_t train_step(const Tensor4D& input, const Tensor4D& target);
    real_t evaluate(const Tensor4D& input, const Tensor4D& target);
    
    void save_weights(const std::string& filename) const;
    bool load_weights(const std::string& filename);
    
    void print_model_info() const;
};