#include "gpu_autoencoder.h"
#include "../common/utils.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

GPUAutoencoder::GPUAutoencoder(real_t lr) : learning_rate(lr) {
    std::cout << "Initializing GPU Autoencoder..." << std::endl;
    init_architecture();
    std::cout << "GPU Autoencoder initialized successfully" << std::endl;
}

GPUAutoencoder::~GPUAutoencoder() {
    // GPU memory cleanup handled automatically by GPUMemory destructors
}

void GPUAutoencoder::init_architecture() {
    // Clear existing configs
    layer_configs.clear();
    
    // Layer 1: Input (32,32,3) -> Conv(256) -> (32,32,256) -> MaxPool -> (16,16,256)
    layer_configs.push_back({3, 256, 3, 1, 1, 32, 32, 32, 32});      // Conv1
    layer_configs.push_back({256, 256, 2, 2, 0, 32, 32, 16, 16});    // MaxPool1
    
    // Layer 2: (16,16,256) -> Conv(128) -> (16,16,128) -> MaxPool -> (8,8,128)
    layer_configs.push_back({256, 128, 3, 1, 1, 16, 16, 16, 16});    // Conv2
    layer_configs.push_back({128, 128, 2, 2, 0, 16, 16, 8, 8});      // MaxPool2
    
    // Decoder layers (mirror encoder)
    // Layer 3: (8,8,128) -> Conv(128) -> (8,8,128) -> Upsample -> (16,16,128)
    layer_configs.push_back({128, 128, 3, 1, 1, 8, 8, 8, 8});        // Conv3
    layer_configs.push_back({128, 128, 2, 1, 0, 8, 8, 16, 16});      // Upsample1
    
    // Layer 4: (16,16,128) -> Conv(256) -> (16,16,256) -> Upsample -> (32,32,256)
    layer_configs.push_back({128, 256, 3, 1, 1, 16, 16, 16, 16});    // Conv4
    layer_configs.push_back({256, 256, 2, 1, 0, 16, 16, 32, 32});    // Upsample2
    
    // Layer 5: (32,32,256) -> Conv(3) -> (32,32,3)
    layer_configs.push_back({256, 3, 3, 1, 1, 32, 32, 32, 32});      // Conv5 (output)
    
    std::cout << "Architecture initialized with " << layer_configs.size() << " layers" << std::endl;
}

void GPUAutoencoder::init_gpu_memory(int batch_size) {
    // Calculate memory requirements
    d_weights.clear();
    d_bias.clear();
    d_activations.clear();
    d_pool_indices.clear();
    
    d_weights.resize(5); // 5 conv layers
    d_bias.resize(5);
    d_activations.resize(layer_configs.size() + 1); // +1 for input
    d_pool_indices.resize(2); // 2 pooling layers
    
    // Allocate weights and biases
    std::vector<int> conv_layers = {0, 2, 4, 6, 8}; // Conv layer indices
    for (int i = 0; i < 5; i++) {
        const auto& config = layer_configs[conv_layers[i]];
        int weight_size = config.output_channels * config.input_channels * 
                         config.kernel_size * config.kernel_size;
        d_weights[i].allocate(weight_size);
        d_bias[i].allocate(config.output_channels);
    }
    
    // Allocate activation buffers
    int max_activation_size = batch_size * 256 * 32 * 32; // Largest activation
    for (auto& activation : d_activations) {
        activation.allocate(max_activation_size);
    }
    
    // Allocate pooling indices
    for (auto& indices : d_pool_indices) {
        indices.allocate(batch_size * 256 * 16 * 16); // Max pooled size
    }
    
    // Allocate input/output buffers
    d_input.allocate(batch_size * 3 * 32 * 32);
    d_output.allocate(batch_size * 3 * 32 * 32);
    d_target.allocate(batch_size * 3 * 32 * 32);
    
    init_weights();
}

void GPUAutoencoder::init_weights() {
    // Initialize weights on CPU then copy to GPU
    for (int i = 0; i < 5; i++) {
        std::vector<int> conv_layers = {0, 2, 4, 6, 8};
        const auto& config = layer_configs[conv_layers[i]];
        
        int weight_size = config.output_channels * config.input_channels * 
                         config.kernel_size * config.kernel_size;
        
        std::vector<float> weights(weight_size);
        std::vector<float> bias(config.output_channels, 0.0f);
        
        // Xavier initialization
        float std_dev = sqrtf(2.0f / (config.input_channels * config.kernel_size * config.kernel_size));
        for (auto& w : weights) {
            w = utils::random_normal() * std_dev;
        }
        
        d_weights[i].copyFromHost(weights.data(), weight_size);
        d_bias[i].copyFromHost(bias.data(), config.output_channels);
    }
}

void GPUAutoencoder::forward(const Tensor4D& input, Tensor4D& output) {
    int batch_size = input.batch_size;
    
    // Initialize GPU memory if needed
    static int last_batch_size = 0;
    if (last_batch_size != batch_size) {
        init_gpu_memory(batch_size);
        last_batch_size = batch_size;
    }
    
    // Copy input to GPU
    d_input.copyFromHost(input.data.data(), input.data.size());
    
    // Forward pass through encoder
    float* current_input = d_input.get();
    float* current_output = d_activations[0].get();
    
    std::vector<int> conv_indices = {0, 2, 4, 6, 8};
    std::vector<int> pool_indices = {1, 3};
    std::vector<int> upsample_indices = {5, 7};
    
    // Conv1 + MaxPool1
    conv2d_forward_gpu(
        current_input, current_output,
        d_weights[0].get(), d_bias[0].get(),
        batch_size, 3, 256, 32, 32, 32, 32, 3, 1, 1
    );
    relu_forward_gpu(current_output, current_output, batch_size * 256 * 32 * 32);
    
    current_input = current_output;
    current_output = d_activations[1].get();
    maxpool2d_forward_gpu(
        current_input, current_output, d_pool_indices[0].get(),
        batch_size, 256, 32, 32, 16, 16, 2, 2
    );
    
    // Conv2 + MaxPool2  
    current_input = current_output;
    current_output = d_activations[2].get();
    conv2d_forward_gpu(
        current_input, current_output,
        d_weights[1].get(), d_bias[1].get(),
        batch_size, 256, 128, 16, 16, 16, 16, 3, 1, 1
    );
    relu_forward_gpu(current_output, current_output, batch_size * 128 * 16 * 16);
    
    current_input = current_output;
    current_output = d_activations[3].get(); // This is the encoded representation
    maxpool2d_forward_gpu(
        current_input, current_output, d_pool_indices[1].get(),
        batch_size, 128, 16, 16, 8, 8, 2, 2
    );
    
    // Decoder
    // Conv3 + Upsample1
    current_input = current_output;
    current_output = d_activations[4].get();
    conv2d_forward_gpu(
        current_input, current_output,
        d_weights[2].get(), d_bias[2].get(),
        batch_size, 128, 128, 8, 8, 8, 8, 3, 1, 1
    );
    relu_forward_gpu(current_output, current_output, batch_size * 128 * 8 * 8);
    
    current_input = current_output;
    current_output = d_activations[5].get();
    upsample2d_forward_gpu(
        current_input, current_output,
        batch_size, 128, 8, 8, 16, 16, 2
    );
    
    // Conv4 + Upsample2
    current_input = current_output;
    current_output = d_activations[6].get();
    conv2d_forward_gpu(
        current_input, current_output,
        d_weights[3].get(), d_bias[3].get(),
        batch_size, 128, 256, 16, 16, 16, 16, 3, 1, 1
    );
    relu_forward_gpu(current_output, current_output, batch_size * 256 * 16 * 16);
    
    current_input = current_output;
    current_output = d_activations[7].get();
    upsample2d_forward_gpu(
        current_input, current_output,
        batch_size, 256, 16, 16, 32, 32, 2
    );
    
    // Conv5 (output layer)
    current_input = current_output;
    conv2d_forward_gpu(
        current_input, d_output.get(),
        d_weights[4].get(), d_bias[4].get(),
        batch_size, 256, 3, 32, 32, 32, 32, 3, 1, 1
    );
    // No ReLU on output layer
    
    // Copy output back to host
    output.resize(batch_size, 32, 32, 3);
    d_output.copyToHost(output.data.data(), output.data.size());
}

void GPUAutoencoder::encode(const Tensor4D& input, Tensor4D& encoded) {
    int batch_size = input.batch_size;
    
    // Initialize GPU memory if needed
    static int last_batch_size = 0;
    if (last_batch_size != batch_size) {
        init_gpu_memory(batch_size);
        last_batch_size = batch_size;
    }
    
    // Copy input to GPU
    d_input.copyFromHost(input.data.data(), input.data.size());
    
    // Forward pass through encoder only
    float* current_input = d_input.get();
    float* current_output = d_activations[0].get();
    
    // Conv1 + MaxPool1
    conv2d_forward_gpu(
        current_input, current_output,
        d_weights[0].get(), d_bias[0].get(),
        batch_size, 3, 256, 32, 32, 32, 32, 3, 1, 1
    );
    relu_forward_gpu(current_output, current_output, batch_size * 256 * 32 * 32);
    
    current_input = current_output;
    current_output = d_activations[1].get();
    maxpool2d_forward_gpu(
        current_input, current_output, d_pool_indices[0].get(),
        batch_size, 256, 32, 32, 16, 16, 2, 2
    );
    
    // Conv2 + MaxPool2  
    current_input = current_output;
    current_output = d_activations[2].get();
    conv2d_forward_gpu(
        current_input, current_output,
        d_weights[1].get(), d_bias[1].get(),
        batch_size, 256, 128, 16, 16, 16, 16, 3, 1, 1
    );
    relu_forward_gpu(current_output, current_output, batch_size * 128 * 16 * 16);
    
    current_input = current_output;
    current_output = d_activations[3].get(); // Encoded representation
    maxpool2d_forward_gpu(
        current_input, current_output, d_pool_indices[1].get(),
        batch_size, 128, 16, 16, 8, 8, 2, 2
    );
    
    // Copy encoded result back to host
    encoded.resize(batch_size, 8, 8, 128);
    d_activations[3].copyToHost(encoded.data.data(), encoded.data.size());
}

real_t GPUAutoencoder::train_step(const Tensor4D& input, const Tensor4D& target) {
    // Forward pass
    Tensor4D output;
    forward(input, output);
    
    // Copy target to GPU
    d_target.copyFromHost(target.data.data(), target.data.size());
    
    // Compute loss
    float loss = mse_loss_gpu(d_output.get(), d_target.get(), output.data.size());
    
    // For now, just return the loss (backward pass would be implemented here)
    return loss;
}

real_t GPUAutoencoder::evaluate(const Tensor4D& input, const Tensor4D& target) {
    // Forward pass
    Tensor4D output;
    forward(input, output);
    
    // Copy target to GPU
    d_target.copyFromHost(target.data.data(), target.data.size());
    
    // Compute loss
    float loss = mse_loss_gpu(d_output.get(), d_target.get(), output.data.size());
    
    return loss;
}

void GPUAutoencoder::save_weights(const std::string& filename) const {
    // Copy weights from GPU to CPU and save
    std::vector<ConvWeights> all_weights(5);
    
    std::vector<int> conv_layers = {0, 2, 4, 6, 8};
    for (int i = 0; i < 5; i++) {
        const auto& config = layer_configs[conv_layers[i]];
        
        int weight_size = config.output_channels * config.input_channels * 
                         config.kernel_size * config.kernel_size;
        
        all_weights[i].output_channels = config.output_channels;
        all_weights[i].input_channels = config.input_channels;
        all_weights[i].kernel_size = config.kernel_size;
        
        all_weights[i].weights.resize(weight_size);
        all_weights[i].bias.resize(config.output_channels);
        
        d_weights[i].copyToHost(all_weights[i].weights.data(), weight_size);
        d_bias[i].copyToHost(all_weights[i].bias.data(), config.output_channels);
    }
    
    utils::save_weights(filename, all_weights);
    std::cout << "GPU weights saved to: " << filename << std::endl;
}

bool GPUAutoencoder::load_weights(const std::string& filename) {
    std::vector<ConvWeights> all_weights;
    
    if (!utils::load_weights(filename, all_weights)) {
        std::cerr << "Failed to load weights from: " << filename << std::endl;
        return false;
    }
    
    if (all_weights.size() != 5) {
        std::cerr << "Expected 5 layers, got " << all_weights.size() << std::endl;
        return false;
    }
    
    // Copy weights from CPU to GPU
    for (int i = 0; i < 5; i++) {
        d_weights[i].copyFromHost(all_weights[i].weights.data(), all_weights[i].weights.size());
        d_bias[i].copyFromHost(all_weights[i].bias.data(), all_weights[i].bias.size());
    }
    
    std::cout << "GPU weights loaded from: " << filename << std::endl;
    return true;
}

void GPUAutoencoder::print_model_info() const {
    std::cout << "\n=== GPU Autoencoder Architecture ===" << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Total Layers: " << layer_configs.size() << std::endl;
    std::cout << "Encoded Size: 8x8x128 = 8,192 features" << std::endl;
    std::cout << "==================================" << std::endl;
}