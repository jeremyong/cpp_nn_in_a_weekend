#pragma once

#include "Model.hpp"

#include <cstdint>
#include <vector>

// Fully-connected, feedforward Layer

// A feedforward layer is parameterized by the number of neurons it posesses and
// the number of neurons in the layer preceding it
class FFNode : public Node
{
public:
    FFNode(Model& model,
           std::string name,
           Activation activation,
           uint16_t output_size,
           uint16_t input_size);

    // Initialize the parameters of the layer
    // F.T.R.
    // Experiment with alternative weight and bias initialization schemes:
    // 1. Try different distributions for the weight
    // 2. Try initializing all weights to zero (why is this suboptimal)
    // 3. Try initializing all the biases to zero
    void init(rne_t& rne) override;

    // The input vector should have size input_size_
    void forward(num_t* inputs) override;
    // The output vector should have size output_size_
    void reverse(num_t* gradients) override;

    size_t param_count() const noexcept override
    {
        // Weight matrix entries + bias entries
        return (input_size_ + 1) * output_size_;
    }

    num_t* param(size_t index);
    num_t* gradient(size_t index);

    void print() const override;

private:
    Activation activation_;
    uint16_t output_size_;
    uint16_t input_size_;

    /////////////////////
    // Node Parameters //
    /////////////////////

    // weights_.size() := output_size_ * input_size_
    std::vector<num_t> weights_;
    // biases_.size() := output_size_
    std::vector<num_t> biases_;
    // activations_.size() := output_size_
    std::vector<num_t> activations_;

    ////////////////////
    // Loss Gradients //
    ////////////////////

    std::vector<num_t> activation_gradients_;

    // During the training cycle, parameter loss gradients are accumulated in
    // the following buffers.
    std::vector<num_t> weight_gradients_;
    std::vector<num_t> bias_gradients_;

    // This buffer is used to store temporary gradients used in a SINGLE
    // backpropagation pass. Note that this does not accumulate like the weight
    // and bias gradients do.
    std::vector<num_t> input_gradients_;

    // The last input is needed to compute loss gradients with respect to the
    // weights during backpropagation
    num_t* last_input_;
};
