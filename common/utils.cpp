#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

void ConvWeights::initialize()
{
    Random rng;
    real_t fan_in = input_channels * kernel_size * kernel_size;
    real_t fan_out = output_channels * kernel_size * kernel_size;

    // Xavier initialization
    real_t limit = sqrt(6.0f / (fan_in + fan_out));

    for (auto &w : weights)
    {
        w = rng.uniform(-limit, limit);
    }

    // Initialize bias to zero
    std::fill(bias.begin(), bias.end(), 0.0f);
}

namespace utils
{
    real_t mse_loss(const Tensor4D &predicted, const Tensor4D &target)
    {
        if (predicted.size() != target.size())
        {
            throw std::runtime_error("Tensor sizes don't match for MSE calculation");
        }

        real_t sum = 0.0f;
        for (size_t i = 0; i < predicted.size(); ++i)
        {
            real_t diff = predicted.data[i] - target.data[i];
            sum += diff * diff;
        }
        return sum / predicted.size();
    }

    void normalize_tensor(Tensor4D &tensor)
    {
        for (auto &val : tensor.data)
        {
            val = val / 255.0f; // Normalize from [0, 255] to [0, 1]
        }
    }

    void save_weights(const std::string &filename, const std::vector<ConvWeights> &weights)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        size_t num_layers = weights.size();
        file.write(reinterpret_cast<const char *>(&num_layers), sizeof(num_layers));

        for (const auto &layer : weights)
        {
            file.write(reinterpret_cast<const char *>(&layer.output_channels), sizeof(int));
            file.write(reinterpret_cast<const char *>(&layer.input_channels), sizeof(int));
            file.write(reinterpret_cast<const char *>(&layer.kernel_size), sizeof(int));

            file.write(reinterpret_cast<const char *>(layer.weights.data()),
                       layer.weights.size() * sizeof(real_t));
            file.write(reinterpret_cast<const char *>(layer.bias.data()),
                       layer.bias.size() * sizeof(real_t));
        }
    }

    bool load_weights(const std::string &filename, std::vector<ConvWeights> &weights)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
        {
            return false;
        }

        size_t num_layers;
        file.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));

        weights.resize(num_layers);

        for (auto &layer : weights)
        {
            file.read(reinterpret_cast<char *>(&layer.output_channels), sizeof(int));
            file.read(reinterpret_cast<char *>(&layer.input_channels), sizeof(int));
            file.read(reinterpret_cast<char *>(&layer.kernel_size), sizeof(int));

            layer.weights.resize(layer.output_channels * layer.input_channels *
                                 layer.kernel_size * layer.kernel_size);
            layer.bias.resize(layer.output_channels);

            file.read(reinterpret_cast<char *>(layer.weights.data()),
                      layer.weights.size() * sizeof(real_t));
            file.read(reinterpret_cast<char *>(layer.bias.data()),
                      layer.bias.size() * sizeof(real_t));
        }

        return true;
    }

    void print_tensor_stats(const Tensor4D &tensor, const std::string &name)
    {
        if (tensor.size() == 0)
        {
            std::cout << name << ": Empty tensor" << std::endl;
            return;
        }

        real_t min_val = *std::min_element(tensor.data.begin(), tensor.data.end());
        real_t max_val = *std::max_element(tensor.data.begin(), tensor.data.end());

        real_t sum = 0.0f;
        for (const auto &val : tensor.data)
        {
            sum += val;
        }
        real_t mean = sum / tensor.size();

        std::cout << name << " - Shape: [" << tensor.batch_size << ", "
                  << tensor.height << ", " << tensor.width << ", "
                  << tensor.channels << "], Min: " << min_val
                  << ", Max: " << max_val << ", Mean: " << mean << std::endl;
    }
}