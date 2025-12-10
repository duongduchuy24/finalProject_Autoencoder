/**
 * @file autoencoder.h
 * @brief CPU-based CIFAR-10 Autoencoder implementation
 * @details Phase 1 baseline with encoder-decoder architecture
 *
 * Architecture Overview:
 * - Input: 32x32x3 RGB images
 * - Encoder: Conv2D + ReLU + MaxPool layers (3->256->128 channels)
 * - Latent: 8x8x128 = 8,192 features
 * - Decoder: Conv2D + ReLU + Upsampling layers (128->256->3 channels)
 * - Output: 32x32x3 reconstructed images
 * - Total Parameters: ~751,875 trainable parameters
 */

#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "../common/types.h"
#include "cnn_layers.h"
#include "../common/utils.h"
#include <memory>
#include <vector>

/**
 * @brief CPU implementation of CIFAR-10 Autoencoder
 * @details Complete encoder-decoder architecture for image reconstruction
 *
 * Key Features:
 * - Encoder: Compresses 32x32x3 -> 8x8x128
 * - Decoder: Reconstructs 8x8x128 -> 32x32x3
 * - MSE Loss for reconstruction training
 * - Feature extraction capability for SVM
 * - Smart pointer memory management
 */
class CPUAutoencoder
{
private:
    //=========================================================================
    // ENCODER LAYERS - Image compression pathway
    //=========================================================================
    std::unique_ptr<cpu_layers::Conv2D> conv1; // 3 -> 256 channels
    std::unique_ptr<cpu_layers::ReLU> relu1;
    std::unique_ptr<cpu_layers::MaxPool2D> pool1; // 32x32 -> 16x16

    std::unique_ptr<cpu_layers::Conv2D> conv2; // 256 -> 128 channels
    std::unique_ptr<cpu_layers::ReLU> relu2;
    std::unique_ptr<cpu_layers::MaxPool2D> pool2; // 16x16 -> 8x8 (LATENT)

    //=========================================================================
    // DECODER LAYERS - Image reconstruction pathway
    //=========================================================================
    std::unique_ptr<cpu_layers::Conv2D> conv3; // 128 -> 128 channels
    std::unique_ptr<cpu_layers::ReLU> relu3;
    std::unique_ptr<cpu_layers::Upsampling2D> up1; // 8x8 -> 16x16

    std::unique_ptr<cpu_layers::Conv2D> conv4; // 128 -> 256 channels
    std::unique_ptr<cpu_layers::ReLU> relu4;
    std::unique_ptr<cpu_layers::Upsampling2D> up2; // 16x16 -> 32x32

    std::unique_ptr<cpu_layers::Conv2D> conv5; // 256 -> 3 channels (RGB output)

    //=========================================================================
    // TRAINING COMPONENTS
    //=========================================================================
    std::unique_ptr<cpu_layers::MSELoss> loss_fn; // Mean squared error loss
    std::vector<Tensor4D> activations;            // Intermediate activations storage
    real_t learning_rate;                         // SGD learning rate

public:
    /**
     * @brief Constructor - Initialize complete autoencoder architecture
     * @param lr Learning rate for training (default: 0.001)
     */
    CPUAutoencoder(real_t lr = 0.001f);

    /**
     * @brief Destructor - Automatic cleanup via smart pointers
     */
    ~CPUAutoencoder();

    //=========================================================================
    // CORE FUNCTIONALITY
    //=========================================================================

    /**
     * @brief Complete forward pass (encode + decode)
     * @param input Input batch [batch_size, 32, 32, 3]
     * @param output Reconstructed output [batch_size, 32, 32, 3]
     */
    void forward(const Tensor4D &input, Tensor4D &output);

    /**
     * @brief Encoder-only forward pass for feature extraction
     * @param input Input batch [batch_size, 32, 32, 3]
     * @param encoded Latent features [batch_size, 8, 8, 128]
     */
    void encode(const Tensor4D &input, Tensor4D &encoded);

    // Backward pass and weight update
    real_t train_step(const Tensor4D &input, const Tensor4D &target);

    // Evaluation (forward pass + loss calculation, no weight update)
    real_t evaluate(const Tensor4D &input, const Tensor4D &target);

    // Weight management
    void save_weights(const std::string &filename) const;
    bool load_weights(const std::string &filename);

    // Utility functions
    void print_architecture() const;
    size_t get_num_parameters() const;

    // Set learning rate
    void set_learning_rate(real_t lr) { learning_rate = lr; }
    real_t get_learning_rate() const { return learning_rate; }

private:
    // Helper function for backward pass
    void backward(const Tensor4D &input, const Tensor4D &target);

    // Simple SGD weight update
    void update_weights();

    // Initialize all activations tensors
    void initialize_activations(int batch_size);
};

#endif // AUTOENCODER_H