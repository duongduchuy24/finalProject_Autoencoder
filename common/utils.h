#ifndef UTILS_H
#define UTILS_H

#include "types.h"
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

//=============================================================================
// PERFORMANCE TIMING
//=============================================================================

class Timer
{
private:
    std::chrono::high_resolution_clock::time_point start_time; ///< Timer start point

public:
    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

class Random
{
private:
    std::mt19937 generator;

public:
    Random(unsigned int seed = 42) : generator(seed) {}

    // Generate random float in range [min, max]
    real_t uniform(real_t min = 0.0f, real_t max = 1.0f)
    {
        std::uniform_real_distribution<real_t> dist(min, max);
        return dist(generator);
    }

    // Generate random normal distribution
    real_t normal(real_t mean = 0.0f, real_t std = 1.0f)
    {
        std::normal_distribution<real_t> dist(mean, std);
        return dist(generator);
    }

    // Shuffle vector indices
    void shuffle(std::vector<int> &indices)
    {
        std::shuffle(indices.begin(), indices.end(), generator);
    }
};

// Utility functions
namespace utils
{
    // Calculate mean squared error
    real_t mse_loss(const Tensor4D &predicted, const Tensor4D &target);

    // Normalize tensor values to [0, 1]
    void normalize_tensor(Tensor4D &tensor);

    // Save/Load model weights
    void save_weights(const std::string &filename, const std::vector<ConvWeights> &weights);
    bool load_weights(const std::string &filename, std::vector<ConvWeights> &weights);

    // Print tensor statistics
    void print_tensor_stats(const Tensor4D &tensor, const std::string &name);
}

#endif // UTILS_H