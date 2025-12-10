#ifndef CNN_LAYERS_H
#define CNN_LAYERS_H

#include "../common/types.h"

namespace cpu_layers
{

    class Conv2D
    {
    private:
        ConvWeights weights;
        int stride;
        int padding;
        int input_channels;
        int output_channels;
        int kernel_size;

    public:
        Conv2D(int input_channels, int output_channels, int kernel_size = 3,
               int stride = 1, int padding = 1);

        // Forward pass
        void forward(const Tensor4D &input, Tensor4D &output);

        // Backward pass
        void backward(const Tensor4D &input, const Tensor4D &grad_output,
                      Tensor4D &grad_input, Tensor4D &grad_weights, Tensor4D &grad_bias);

        // Weight access
        ConvWeights &get_weights() { return weights; }
        const ConvWeights &get_weights() const { return weights; }

        // Layer information
        int get_input_channels() const { return input_channels; }
        int get_output_channels() const { return output_channels; }
        int get_kernel_size() const { return kernel_size; }
        int get_stride() const { return stride; }
        int get_padding() const { return padding; }

        // Calculate output dimensions
        void calculate_output_size(int input_h, int input_w, int &output_h, int &output_w) const;

        // Parameter count
        size_t parameter_count() const;

        // Initialize weights with different strategies
        void initialize_weights(const std::string &method = "xavier");
    };

    class ReLU
    {
    public:
        ReLU() = default;

        // Forward pass: output = max(0, input)
        void forward(const Tensor4D &input, Tensor4D &output);

        // Backward pass: gradient flows through if input > 0
        void backward(const Tensor4D &input, const Tensor4D &grad_output, Tensor4D &grad_input);
    };

    class MaxPool2D
    {
    private:
        int pool_size;
        int stride;

    public:
        MaxPool2D(int pool_size = 2, int stride = 2);

        // Forward pass
        void forward(const Tensor4D &input, Tensor4D &output);

        // Backward pass
        void backward(const Tensor4D &input, const Tensor4D &output,
                      const Tensor4D &grad_output, Tensor4D &grad_input);

        // Calculate output dimensions
        void calculate_output_size(int input_h, int input_w, int &output_h, int &output_w) const;
    };

    class Upsampling2D
    {
    private:
        int scale_factor;

    public:
        Upsampling2D(int scale_factor = 2);

        // Forward pass (nearest neighbor interpolation)
        void forward(const Tensor4D &input, Tensor4D &output);

        // Backward pass
        void backward(const Tensor4D &grad_output, Tensor4D &grad_input);

        // Calculate output dimensions
        void calculate_output_size(int input_h, int input_w, int &output_h, int &output_w) const;
    };

    // Loss function
    class MSELoss
    {
    public:
        MSELoss() = default;

        // Calculate mean squared error loss
        real_t forward(const Tensor4D &predicted, const Tensor4D &target);

        // Calculate gradients
        void backward(const Tensor4D &predicted, const Tensor4D &target, Tensor4D &grad_input);
    };

} // namespace cpu_layers

#endif // CNN_LAYERS_H