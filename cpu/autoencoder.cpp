/**
 * @file autoencoder.cpp
 * @brief CPU implementation of CIFAR-10 Autoencoder
 * @details Complete encoder-decoder architecture with 751K+ parameters
 * Designed for Phase 1 baseline implementation
 */

#include "autoencoder.h"
#include "../common/utils.h"
#include <iostream>
#include <fstream>

/**
 * @brief Constructor - Builds complete autoencoder architecture
 * @param lr Learning rate for training
 *
 * Architecture Summary:
 * ENCODER: 32x32x3 -> Conv(256) -> ReLU -> Pool -> Conv(128) -> ReLU -> Pool -> 8x8x128
 * DECODER: 8x8x128 -> Conv(128) -> ReLU -> Up -> Conv(256) -> ReLU -> Up -> Conv(3) -> 32x32x3
 * Total Parameters: ~751K (trainable weights + biases)
 */
CPUAutoencoder::CPUAutoencoder(real_t lr) : learning_rate(lr)
{
    //=========================================================================
    // ENCODER LAYERS - Compress input to latent representation
    //=========================================================================
    conv1 = std::make_unique<cpu_layers::Conv2D>(3, 256, 3, 1, 1); // 32x32x3 -> 32x32x256
    relu1 = std::make_unique<cpu_layers::ReLU>();
    pool1 = std::make_unique<cpu_layers::MaxPool2D>(2, 2); // 32x32x256 -> 16x16x256

    conv2 = std::make_unique<cpu_layers::Conv2D>(256, 128, 3, 1, 1); // 16x16x256 -> 16x16x128
    relu2 = std::make_unique<cpu_layers::ReLU>();
    pool2 = std::make_unique<cpu_layers::MaxPool2D>(2, 2); // 16x16x128 -> 8x8x128 (LATENT)

    //=========================================================================
    // DECODER LAYERS - Reconstruct from latent representation
    //=========================================================================
    conv3 = std::make_unique<cpu_layers::Conv2D>(128, 128, 3, 1, 1); // 8x8x128 -> 8x8x128
    relu3 = std::make_unique<cpu_layers::ReLU>();
    up1 = std::make_unique<cpu_layers::Upsampling2D>(2); // 8x8x128 -> 16x16x128

    conv4 = std::make_unique<cpu_layers::Conv2D>(128, 256, 3, 1, 1); // 16x16x128 -> 16x16x256
    relu4 = std::make_unique<cpu_layers::ReLU>();
    up2 = std::make_unique<cpu_layers::Upsampling2D>(2); // 16x16x256 -> 32x32x256

    conv5 = std::make_unique<cpu_layers::Conv2D>(256, 3, 3, 1, 1); // 32x32x256 -> 32x32x3 (OUTPUT)

    //=========================================================================
    // LOSS FUNCTION AND ACTIVATIONS
    //=========================================================================
    loss_fn = std::make_unique<cpu_layers::MSELoss>();
    activations.resize(11); // Pre-allocate for intermediate outputs

    // Display architecture information
    std::cout << "CPUAutoencoder initialized with " << get_num_parameters() << " parameters" << std::endl;
    print_architecture();
}

/**
 * @brief Destructor - Smart pointers handle automatic cleanup
 */
CPUAutoencoder::~CPUAutoencoder()
{
    // All unique_ptr members are automatically cleaned up
}

/**
 * @brief Complete forward pass through autoencoder
 * @param input Input batch [batch_size, 32, 32, 3]
 * @param output Reconstructed output [batch_size, 32, 32, 3]
 *
 * Flow: Input -> Encoder -> Latent (8x8x128) -> Decoder -> Output
 * Stores all intermediate activations for potential backward pass
 */
void CPUAutoencoder::forward(const Tensor4D &input, Tensor4D &output)
{
    // Prepare activation storage for this batch
    initialize_activations(input.batch_size);

    //=========================================================================
    // ENCODER: Compress input to latent representation
    //=========================================================================
    conv1->forward(input, activations[0]);          // 32x32x3 -> 32x32x256
    relu1->forward(activations[0], activations[1]); // Apply ReLU activation
    pool1->forward(activations[1], activations[2]); // 32x32x256 -> 16x16x256

    conv2->forward(activations[2], activations[3]); // 16x16x256 -> 16x16x128
    relu2->forward(activations[3], activations[4]); // ReLU activation
    pool2->forward(activations[4], activations[5]); // 16x16x128 -> 8x8x128 (latent)

    // Decoder forward pass
    conv3->forward(activations[5], activations[6]); // 8x8x128 -> 8x8x128
    relu3->forward(activations[6], activations[7]); // ReLU activation
    up1->forward(activations[7], activations[8]);   // 8x8x128 -> 16x16x128

    conv4->forward(activations[8], activations[9]);  // 16x16x128 -> 16x16x256
    relu4->forward(activations[9], activations[10]); // ReLU activation
    up2->forward(activations[10], output);           // 16x16x256 -> 32x32x256

    conv5->forward(output, output); // 32x32x256 -> 32x32x3 (final output)
}

void CPUAutoencoder::encode(const Tensor4D &input, Tensor4D &encoded)
{
    initialize_activations(input.batch_size);

    // Encoder forward pass only
    conv1->forward(input, activations[0]);
    relu1->forward(activations[0], activations[1]);
    pool1->forward(activations[1], activations[2]);

    conv2->forward(activations[2], activations[3]);
    relu2->forward(activations[3], activations[4]);
    pool2->forward(activations[4], encoded); // Output latent representation (8x8x128)
}

real_t CPUAutoencoder::train_step(const Tensor4D &input, const Tensor4D &target)
{
    // Forward pass
    Tensor4D output;
    forward(input, output);

    // Calculate loss
    real_t loss = loss_fn->forward(output, target);

    // Backward pass
    backward(input, target);

    // Update weights
    update_weights();

    return loss;
}

real_t CPUAutoencoder::evaluate(const Tensor4D &input, const Tensor4D &target)
{
    Tensor4D output;
    forward(input, output);
    return loss_fn->forward(output, target);
}

void CPUAutoencoder::backward(const Tensor4D &input, const Tensor4D &target)
{
    // This is a simplified backward pass implementation
    // In a full implementation, you would need to compute gradients for all layers
    // For now, we'll implement a basic version that shows the structure

    std::cout << "Backward pass - simplified implementation" << std::endl;
    // TODO: Implement full gradient computation
}

void CPUAutoencoder::update_weights()
{
    // This is a simplified weight update implementation
    // In a full implementation, you would update weights based on computed gradients

    std::cout << "Weight update - simplified implementation" << std::endl;
    // TODO: Implement full weight update using gradients
}

void CPUAutoencoder::save_weights(const std::string &filename) const
{
    std::vector<ConvWeights> all_weights;

    all_weights.push_back(conv1->get_weights());
    all_weights.push_back(conv2->get_weights());
    all_weights.push_back(conv3->get_weights());
    all_weights.push_back(conv4->get_weights());
    all_weights.push_back(conv5->get_weights());

    utils::save_weights(filename, all_weights);
    std::cout << "Weights saved to: " << filename << std::endl;
}

bool CPUAutoencoder::load_weights(const std::string &filename)
{
    std::vector<ConvWeights> all_weights;

    if (!utils::load_weights(filename, all_weights))
    {
        std::cerr << "Failed to load weights from: " << filename << std::endl;
        return false;
    }

    if (all_weights.size() != 5)
    {
        std::cerr << "Expected 5 layers, got " << all_weights.size() << std::endl;
        return false;
    }

    conv1->get_weights() = all_weights[0];
    conv2->get_weights() = all_weights[1];
    conv3->get_weights() = all_weights[2];
    conv4->get_weights() = all_weights[3];
    conv5->get_weights() = all_weights[4];

    std::cout << "Weights loaded from: " << filename << std::endl;
    return true;
}

void CPUAutoencoder::print_architecture() const
{
    std::cout << "\n=== Autoencoder Architecture ===" << std::endl;
    std::cout << "ENCODER:" << std::endl;
    std::cout << "  Conv1: 3 -> 256 channels (3x3, stride=1, padding=1)" << std::endl;
    std::cout << "  ReLU + MaxPool2D (2x2, stride=2): 32x32 -> 16x16" << std::endl;
    std::cout << "  Conv2: 256 -> 128 channels (3x3, stride=1, padding=1)" << std::endl;
    std::cout << "  ReLU + MaxPool2D (2x2, stride=2): 16x16 -> 8x8" << std::endl;
    std::cout << "  LATENT: 8x8x128 = 8192 features" << std::endl;

    std::cout << "\nDECODER:" << std::endl;
    std::cout << "  Conv3: 128 -> 128 channels (3x3, stride=1, padding=1)" << std::endl;
    std::cout << "  ReLU + Upsampling2D (2x): 8x8 -> 16x16" << std::endl;
    std::cout << "  Conv4: 128 -> 256 channels (3x3, stride=1, padding=1)" << std::endl;
    std::cout << "  ReLU + Upsampling2D (2x): 16x16 -> 32x32" << std::endl;
    std::cout << "  Conv5: 256 -> 3 channels (3x3, stride=1, padding=1)" << std::endl;
    std::cout << "  OUTPUT: 32x32x3" << std::endl;
    std::cout << "========================\n"
              << std::endl;
}

size_t CPUAutoencoder::get_num_parameters() const
{
    size_t total_params = 0;

    // Conv1: (3 * 256 * 3 * 3) + 256 bias
    total_params += (3 * 256 * 3 * 3) + 256;

    // Conv2: (256 * 128 * 3 * 3) + 128 bias
    total_params += (256 * 128 * 3 * 3) + 128;

    // Conv3: (128 * 128 * 3 * 3) + 128 bias
    total_params += (128 * 128 * 3 * 3) + 128;

    // Conv4: (128 * 256 * 3 * 3) + 256 bias
    total_params += (128 * 256 * 3 * 3) + 256;

    // Conv5: (256 * 3 * 3 * 3) + 3 bias
    total_params += (256 * 3 * 3 * 3) + 3;

    return total_params;
}

void CPUAutoencoder::initialize_activations(int batch_size)
{
    // Initialize activation tensors with proper sizes
    activations[0].resize(batch_size, 32, 32, 256);  // After conv1
    activations[1].resize(batch_size, 32, 32, 256);  // After relu1
    activations[2].resize(batch_size, 16, 16, 256);  // After pool1
    activations[3].resize(batch_size, 16, 16, 128);  // After conv2
    activations[4].resize(batch_size, 16, 16, 128);  // After relu2
    activations[5].resize(batch_size, 8, 8, 128);    // After pool2 (latent)
    activations[6].resize(batch_size, 8, 8, 128);    // After conv3
    activations[7].resize(batch_size, 8, 8, 128);    // After relu3
    activations[8].resize(batch_size, 16, 16, 128);  // After up1
    activations[9].resize(batch_size, 16, 16, 256);  // After conv4
    activations[10].resize(batch_size, 16, 16, 256); // After relu4
}