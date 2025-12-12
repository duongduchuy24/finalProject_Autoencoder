#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <memory>

typedef float real_t;
typedef unsigned char uchar;

//=============================================================================
// 4D TENSOR STRUCTURE
//=============================================================================
struct Tensor4D
{
    std::vector<real_t> data; ///< Flattened data storage
    int batch_size;           ///< Number of samples in batch
    int height;               ///< Spatial height dimension
    int width;                ///< Spatial width dimension
    int channels;             ///< Channel/feature dimension

    Tensor4D(int b = 0, int h = 0, int w = 0, int c = 0)
        : batch_size(b), height(h), width(w), channels(c)
    {
        data.resize(b * h * w * c, 0.0f);
    }

    real_t &operator()(int b, int h, int w, int c)
    {
        return data[((b * height + h) * width + w) * channels + c];
    }

    const real_t &operator()(int b, int h, int w, int c) const
    {
        return data[((b * height + h) * width + w) * channels + c];
    }

    void zero()
    {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    // Get total size
    size_t size() const
    {
        return data.size();
    }

    // Reshape tensor
    void resize(int b, int h, int w, int c)
    {
        batch_size = b;
        height = h;
        width = w;
        channels = c;
        data.resize(b * h * w * c, 0.0f);
    }
};

// Weights structure for convolutional layers
struct ConvWeights
{
    std::vector<real_t> weights; // [output_channels][input_channels][kernel_height][kernel_width]
    std::vector<real_t> bias;    // [output_channels]
    int output_channels;
    int input_channels;
    int kernel_size;

    ConvWeights(int out_ch = 0, int in_ch = 0, int k_size = 3)
        : output_channels(out_ch), input_channels(in_ch), kernel_size(k_size)
    {
        weights.resize(out_ch * in_ch * k_size * k_size, 0.0f);
        bias.resize(out_ch, 0.0f);
    }

    // Initialize weights with Xavier/He initialization
    void initialize();
};

#endif // TYPES_H