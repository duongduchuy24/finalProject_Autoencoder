#include "cnn_layers.h"
#include "../common/utils.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace cpu_layers
{

    //=============================================================================
    // Conv2D Layer Implementation
    //=============================================================================

    Conv2D::Conv2D(int input_channels, int output_channels, int kernel_size, int stride, int padding)
        : weights(output_channels, input_channels, kernel_size), stride(stride), padding(padding),
          input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size)
    {
        weights.initialize(); // Xavier initialization by default
    }

    void Conv2D::forward(const Tensor4D &input, Tensor4D &output)
    {
        // Extract input dimensions
        const int batch_size = input.batch_size;
        const int input_h = input.height;
        const int input_w = input.width;
        const int input_c = input.channels;

        // Calculate output dimensions
        int output_h, output_w;
        calculate_output_size(input_h, input_w, output_h, output_w);

        // Resize output tensor to correct dimensions
        output.resize(batch_size, output_h, output_w, weights.output_channels);

        // Perform convolution: O[b,oh,ow,oc] = Σ I[b,ih,iw,ic] * W[oc,ic,kh,kw] + B[oc]
        for (int b = 0; b < batch_size; ++b)
        {
            for (int oc = 0; oc < weights.output_channels; ++oc)
            {
                for (int oh = 0; oh < output_h; ++oh)
                {
                    for (int ow = 0; ow < output_w; ++ow)
                    {
                        // Start with bias term
                        real_t sum = weights.bias[oc];

                        // Convolve over all input channels and kernel positions
                        for (int ic = 0; ic < input_c; ++ic)
                        {
                            for (int kh = 0; kh < weights.kernel_size; ++kh)
                            {
                                for (int kw = 0; kw < weights.kernel_size; ++kw)
                                {
                                    // Calculate input position with padding
                                    const int ih = oh * stride + kh - padding;
                                    const int iw = ow * stride + kw - padding;

                                    // Apply zero padding (skip if outside input bounds)
                                    if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w)
                                    {
                                        // Calculate weight index in flattened array
                                        const int weight_idx = ((oc * input_c + ic) * weights.kernel_size + kh) * weights.kernel_size + kw;
                                        sum += input(b, ih, iw, ic) * weights.weights[weight_idx];
                                    }
                                }
                            }
                        }
                        output(b, oh, ow, oc) = sum;
                    }
                }
            }
        }
    }

    void Conv2D::backward(const Tensor4D &input, const Tensor4D &grad_output,
                          Tensor4D &grad_input, Tensor4D &grad_weights, Tensor4D &grad_bias)
    {
        int batch_size = input.batch_size;
        int input_h = input.height;
        int input_w = input.width;
        int input_c = input.channels;
        int output_h = grad_output.height;
        int output_w = grad_output.width;

        // Initialize gradients
        grad_input.resize(batch_size, input_h, input_w, input_c);
        grad_weights.resize(weights.output_channels, input_c, weights.kernel_size, weights.kernel_size);
        grad_bias.resize(1, 1, 1, weights.output_channels);

        // Clear gradients
        std::fill(grad_input.data.begin(), grad_input.data.end(), 0.0f);
        std::fill(grad_weights.data.begin(), grad_weights.data.end(), 0.0f);
        std::fill(grad_bias.data.begin(), grad_bias.data.end(), 0.0f);

        // Compute gradients
        for (int b = 0; b < batch_size; ++b)
        {
            for (int oc = 0; oc < weights.output_channels; ++oc)
            {
                for (int oh = 0; oh < output_h; ++oh)
                {
                    for (int ow = 0; ow < output_w; ++ow)
                    {
                        real_t grad_out_val = grad_output(b, oh, ow, oc);

                        // Gradient w.r.t bias
                        grad_bias(0, 0, 0, oc) += grad_out_val;

                        for (int ic = 0; ic < input_c; ++ic)
                        {
                            for (int kh = 0; kh < weights.kernel_size; ++kh)
                            {
                                for (int kw = 0; kw < weights.kernel_size; ++kw)
                                {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;

                                    if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w)
                                    {
                                        // Gradient w.r.t weights
                                        grad_weights(oc, ic, kh, kw) += grad_out_val * input(b, ih, iw, ic);

                                        // Gradient w.r.t input
                                        int weight_idx = ((oc * input_c + ic) * weights.kernel_size + kh) * weights.kernel_size + kw;
                                        grad_input(b, ih, iw, ic) += grad_out_val * weights.weights[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void Conv2D::calculate_output_size(int input_h, int input_w, int &output_h, int &output_w) const
    {
        output_h = (input_h + 2 * padding - weights.kernel_size) / stride + 1;
        output_w = (input_w + 2 * padding - weights.kernel_size) / stride + 1;
    }

    size_t Conv2D::parameter_count() const
    {
        const size_t weight_params = weights.output_channels * weights.input_channels *
                                     weights.kernel_size * weights.kernel_size;
        const size_t bias_params = weights.output_channels;
        return weight_params + bias_params;
    }

    void Conv2D::initialize_weights(const std::string &method)
    {
        if (method == "xavier" || method == "glorot")
        {
            weights.initialize(); // Default Xavier initialization
        }
        else if (method == "he")
        {
            // He initialization for ReLU networks
            real_t fan_in = input_channels * kernel_size * kernel_size;
            real_t std = std::sqrt(2.0f / fan_in);

            Random rng(std::random_device{}());
            for (size_t i = 0; i < weights.weights.size(); ++i)
            {
                weights.weights[i] = rng.normal(0.0f, std);
            }

            // Bias to zero
            std::fill(weights.bias.begin(), weights.bias.end(), 0.0f);
        }
        else if (method == "zero")
        {
            std::fill(weights.weights.begin(), weights.weights.end(), 0.0f);
            std::fill(weights.bias.begin(), weights.bias.end(), 0.0f);
        }
    }

    //=============================================================================
    // ReLU Activation Layer Implementation
    //=============================================================================
    void ReLU::forward(const Tensor4D &input, Tensor4D &output)
    {
        output.resize(input.batch_size, input.height, input.width, input.channels);

        // Apply ReLU element-wise: output[i] = max(0, input[i])
        const size_t total_elements = input.size();
        for (size_t i = 0; i < total_elements; ++i)
        {
            output.data[i] = std::max(0.0f, input.data[i]);
        }
    }

    void ReLU::backward(const Tensor4D &input, const Tensor4D &grad_output, Tensor4D &grad_input)
    {
        grad_input.resize(input.batch_size, input.height, input.width, input.channels);

        // Apply ReLU derivative element-wise
        const size_t total_elements = input.size();
        for (size_t i = 0; i < total_elements; ++i)
        {
            grad_input.data[i] = (input.data[i] > 0.0f) ? grad_output.data[i] : 0.0f;
        }
    }

    //=============================================================================
    // MaxPool2D Layer Implementation
    //=============================================================================

    MaxPool2D::MaxPool2D(int pool_size, int stride) : pool_size(pool_size), stride(stride)
    {
    }

    void MaxPool2D::forward(const Tensor4D &input, Tensor4D &output)
    {
        int batch_size = input.batch_size;
        int input_h = input.height;
        int input_w = input.width;
        int channels = input.channels;

        int output_h, output_w;
        calculate_output_size(input_h, input_w, output_h, output_w);

        output.resize(batch_size, output_h, output_w, channels);

        for (int b = 0; b < batch_size; ++b)
        {
            for (int c = 0; c < channels; ++c)
            {
                for (int oh = 0; oh < output_h; ++oh)
                {
                    for (int ow = 0; ow < output_w; ++ow)
                    {
                        real_t max_val = -1e6f;

                        for (int ph = 0; ph < pool_size; ++ph)
                        {
                            for (int pw = 0; pw < pool_size; ++pw)
                            {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;

                                if (ih < input_h && iw < input_w)
                                {
                                    max_val = std::max(max_val, input(b, ih, iw, c));
                                }
                            }
                        }

                        output(b, oh, ow, c) = max_val;
                    }
                }
            }
        }
    }

    void MaxPool2D::backward(const Tensor4D &input, const Tensor4D &output,
                             const Tensor4D &grad_output, Tensor4D &grad_input)
    {
        int batch_size = input.batch_size;
        int input_h = input.height;
        int input_w = input.width;
        int channels = input.channels;
        int output_h = output.height;
        int output_w = output.width;

        grad_input.resize(batch_size, input_h, input_w, channels);
        std::fill(grad_input.data.begin(), grad_input.data.end(), 0.0f);

        for (int b = 0; b < batch_size; ++b)
        {
            for (int c = 0; c < channels; ++c)
            {
                for (int oh = 0; oh < output_h; ++oh)
                {
                    for (int ow = 0; ow < output_w; ++ow)
                    {
                        real_t max_val = output(b, oh, ow, c);
                        real_t grad_val = grad_output(b, oh, ow, c);

                        // Find the position of max value and propagate gradient
                        for (int ph = 0; ph < pool_size; ++ph)
                        {
                            for (int pw = 0; pw < pool_size; ++pw)
                            {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;

                                if (ih < input_h && iw < input_w &&
                                    std::abs(input(b, ih, iw, c) - max_val) < 1e-6f)
                                {
                                    grad_input(b, ih, iw, c) += grad_val;
                                    // Break after finding the first max (handles ties)
                                    goto next_output_pixel;
                                }
                            }
                        }
                    next_output_pixel:;
                    }
                }
            }
        }
    }

    void MaxPool2D::calculate_output_size(int input_h, int input_w, int &output_h, int &output_w) const
    {
        output_h = (input_h - pool_size) / stride + 1;
        output_w = (input_w - pool_size) / stride + 1;
    }

    // Upsampling2D Implementation
    Upsampling2D::Upsampling2D(int scale_factor) : scale_factor(scale_factor)
    {
    }

    void Upsampling2D::forward(const Tensor4D &input, Tensor4D &output)
    {
        int batch_size = input.batch_size;
        int input_h = input.height;
        int input_w = input.width;
        int channels = input.channels;

        int output_h, output_w;
        calculate_output_size(input_h, input_w, output_h, output_w);

        output.resize(batch_size, output_h, output_w, channels);

        // Nearest neighbor upsampling
        for (int b = 0; b < batch_size; ++b)
        {
            for (int c = 0; c < channels; ++c)
            {
                for (int oh = 0; oh < output_h; ++oh)
                {
                    for (int ow = 0; ow < output_w; ++ow)
                    {
                        int ih = oh / scale_factor;
                        int iw = ow / scale_factor;

                        output(b, oh, ow, c) = input(b, ih, iw, c);
                    }
                }
            }
        }
    }

    void Upsampling2D::backward(const Tensor4D &grad_output, Tensor4D &grad_input)
    {
        int batch_size = grad_output.batch_size;
        int output_h = grad_output.height;
        int output_w = grad_output.width;
        int channels = grad_output.channels;

        int input_h = output_h / scale_factor;
        int input_w = output_w / scale_factor;

        grad_input.resize(batch_size, input_h, input_w, channels);
        std::fill(grad_input.data.begin(), grad_input.data.end(), 0.0f);

        // Sum gradients from all upsampled positions back to original positions
        for (int b = 0; b < batch_size; ++b)
        {
            for (int c = 0; c < channels; ++c)
            {
                for (int oh = 0; oh < output_h; ++oh)
                {
                    for (int ow = 0; ow < output_w; ++ow)
                    {
                        int ih = oh / scale_factor;
                        int iw = ow / scale_factor;

                        if (ih < input_h && iw < input_w)
                        {
                            grad_input(b, ih, iw, c) += grad_output(b, oh, ow, c);
                        }
                    }
                }
            }
        }
    }

    void Upsampling2D::calculate_output_size(int input_h, int input_w, int &output_h, int &output_w) const
    {
        output_h = input_h * scale_factor;
        output_w = input_w * scale_factor;
    }

    // MSELoss Implementation
    real_t MSELoss::forward(const Tensor4D &predicted, const Tensor4D &target)
    {
        if (predicted.size() != target.size())
        {
            throw std::runtime_error("Predicted and target tensors must have the same size");
        }

        real_t sum = 0.0f;
        for (size_t i = 0; i < predicted.size(); ++i)
        {
            real_t diff = predicted.data[i] - target.data[i];
            sum += diff * diff;
        }

        return sum / predicted.size();
    }

    void MSELoss::backward(const Tensor4D &predicted, const Tensor4D &target, Tensor4D &grad_input)
    {
        grad_input.resize(predicted.batch_size, predicted.height, predicted.width, predicted.channels);

        real_t scale = 2.0f / predicted.size();

        for (size_t i = 0; i < predicted.size(); ++i)
        {
            grad_input.data[i] = scale * (predicted.data[i] - target.data[i]);
        }
    }

} // namespace cpu_layers