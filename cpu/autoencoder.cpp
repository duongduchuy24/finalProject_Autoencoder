#include "autoencoder.h"
#include "../common/utils.h"
#include <iostream>
#include <fstream>

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
    activations.resize(12); // Pre-allocate for intermediate outputs (0-11)

    // Display architecture information
    std::cout << "CPUAutoencoder initialized with " << get_num_parameters() << " parameters" << std::endl;
    print_architecture();
}

CPUAutoencoder::~CPUAutoencoder()
{
    // All unique_ptr members are automatically cleaned up
}

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
    up2->forward(activations[10], activations[11]);  // 16x16x256 -> 32x32x256

    conv5->forward(activations[11], output); // 32x32x256 -> 32x32x3 (final output)
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
    // Forward pass to get output and store intermediate activations
    Tensor4D output;
    forward(input, output);

    // Initialize gradient storage
    std::vector<Tensor4D> gradients(12);   // One for each activation (0-11)
    std::vector<Tensor4D> weight_grads(5); // One for each conv layer
    std::vector<Tensor4D> bias_grads(5);   // One for each conv layer

    // Calculate loss gradient (dL/dOutput)
    Tensor4D loss_grad;
    loss_fn->backward(output, target, loss_grad);

    //=========================================================================
    // BACKWARD PASS THROUGH DECODER
    //=========================================================================

    // Conv5 backward: 32x32x256 <- 32x32x3
    conv5->backward(activations[11], loss_grad, gradients[11], weight_grads[4], bias_grads[4]);

    // Upsampling2 backward: 16x16x256 <- 32x32x256
    up2->backward(gradients[11], gradients[10]);

    // ReLU4 backward: 16x16x256 <- 16x16x256
    relu4->backward(activations[9], gradients[9], gradients[9]);

    // Conv4 backward: 16x16x128 <- 16x16x256
    conv4->backward(activations[8], gradients[9], gradients[8], weight_grads[3], bias_grads[3]);

    // Upsampling1 backward: 8x8x128 <- 16x16x128
    up1->backward(gradients[8], gradients[7]);

    // ReLU3 backward: 8x8x128 <- 8x8x128
    relu3->backward(activations[6], gradients[7], gradients[6]);

    // Conv3 backward: 8x8x128 <- 8x8x128
    conv3->backward(activations[5], gradients[6], gradients[5], weight_grads[2], bias_grads[2]);

    //=========================================================================
    // BACKWARD PASS THROUGH ENCODER
    //=========================================================================

    // MaxPool2 backward: 16x16x128 <- 8x8x128
    pool2->backward(activations[4], activations[5], gradients[5], gradients[4]);

    // ReLU2 backward: 16x16x128 <- 16x16x128
    relu2->backward(activations[3], gradients[4], gradients[3]);

    // Conv2 backward: 16x16x256 <- 16x16x128
    conv2->backward(activations[2], gradients[3], gradients[2], weight_grads[1], bias_grads[1]);

    // MaxPool1 backward: 32x32x256 <- 16x16x256
    pool1->backward(activations[1], activations[2], gradients[2], gradients[1]);

    // ReLU1 backward: 32x32x256 <- 32x32x256
    relu1->backward(activations[0], gradients[1], gradients[0]);

    // Conv1 backward: 32x32x3 <- 32x32x256
    Tensor4D input_grad;
    conv1->backward(input, gradients[0], input_grad, weight_grads[0], bias_grads[0]);

    // Store gradients for weight update
    layer_weight_grads = std::move(weight_grads);
    layer_bias_grads = std::move(bias_grads);
}

void CPUAutoencoder::update_weights()
{
    // Apply gradients to all conv layers using SGD
    // weight = weight - learning_rate * gradient

    if (layer_weight_grads.size() != 5 || layer_bias_grads.size() != 5)
    {
        std::cerr << "Error: Gradients not computed. Call backward() first." << std::endl;
        return;
    }

    //=========================================================================
    // UPDATE CONV LAYER WEIGHTS
    //=========================================================================

    // Update Conv1 weights
    auto &conv1_weights = conv1->get_weights();
    for (size_t i = 0; i < conv1_weights.weights.size(); ++i)
    {
        conv1_weights.weights[i] -= learning_rate * layer_weight_grads[0].data[i];
    }
    for (size_t i = 0; i < conv1_weights.bias.size(); ++i)
    {
        conv1_weights.bias[i] -= learning_rate * layer_bias_grads[0].data[i];
    }

    // Update Conv2 weights
    auto &conv2_weights = conv2->get_weights();
    for (size_t i = 0; i < conv2_weights.weights.size(); ++i)
    {
        conv2_weights.weights[i] -= learning_rate * layer_weight_grads[1].data[i];
    }
    for (size_t i = 0; i < conv2_weights.bias.size(); ++i)
    {
        conv2_weights.bias[i] -= learning_rate * layer_bias_grads[1].data[i];
    }

    // Update Conv3 weights
    auto &conv3_weights = conv3->get_weights();
    for (size_t i = 0; i < conv3_weights.weights.size(); ++i)
    {
        conv3_weights.weights[i] -= learning_rate * layer_weight_grads[2].data[i];
    }
    for (size_t i = 0; i < conv3_weights.bias.size(); ++i)
    {
        conv3_weights.bias[i] -= learning_rate * layer_bias_grads[2].data[i];
    }

    // Update Conv4 weights
    auto &conv4_weights = conv4->get_weights();
    for (size_t i = 0; i < conv4_weights.weights.size(); ++i)
    {
        conv4_weights.weights[i] -= learning_rate * layer_weight_grads[3].data[i];
    }
    for (size_t i = 0; i < conv4_weights.bias.size(); ++i)
    {
        conv4_weights.bias[i] -= learning_rate * layer_bias_grads[3].data[i];
    }

    // Update Conv5 weights
    auto &conv5_weights = conv5->get_weights();
    for (size_t i = 0; i < conv5_weights.weights.size(); ++i)
    {
        conv5_weights.weights[i] -= learning_rate * layer_weight_grads[4].data[i];
    }
    for (size_t i = 0; i < conv5_weights.bias.size(); ++i)
    {
        conv5_weights.bias[i] -= learning_rate * layer_bias_grads[4].data[i];
    }

    // Clear gradients after update
    layer_weight_grads.clear();
    layer_bias_grads.clear();
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
    activations[11].resize(batch_size, 32, 32, 256); // After up2
}