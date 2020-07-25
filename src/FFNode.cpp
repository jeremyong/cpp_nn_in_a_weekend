#include "FFNode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

FFNode::FFNode(Model& model,
               std::string name,
               Activation activation,
               uint16_t output_size,
               uint16_t input_size)
    : Node{model, std::move(name)}
    , activation_{activation}
    , output_size_{output_size}
    , input_size_{input_size}
{
    std::printf("%s: %d -> %d\n", name_.c_str(), input_size_, output_size_);

    // The weight parameters of a FF-layer are an NxM matrix
    weights_.resize(output_size_ * input_size_);

    // Each node in this layer is assigned a bias (so that zero is not
    // necessarily mapped to zero)
    biases_.resize(output_size_);

    // The outputs of each neuron within the layer is an "activation" in
    // neuroscience parlance
    activations_.resize(output_size_);

    activation_gradients_.resize(output_size_);
    weight_gradients_.resize(output_size_ * input_size_);
    bias_gradients_.resize(output_size_);
    input_gradients_.resize(input_size_);
}

void FFNode::init(rne_t& rne)
{
    num_t sigma;
    switch (activation_)
    {
    case Activation::ReLU:
        // Kaiming He, et. al. weight initialization for ReLU networks
        // https://arxiv.org/pdf/1502.01852.pdf
        //
        // Suggests using a normal distribution with variance := 2 / n_in
        sigma = std::sqrt(2.0 / static_cast<num_t>(input_size_));
        break;
    case Activation::Softmax:
    default:
        sigma = std::sqrt(1.0 / static_cast<num_t>(input_size_));
        break;
    }

    // NOTE: Unfortunately, the C++ standard does not guarantee that the results
    // obtained from a distribution function will be identical given the same
    // inputs across different compilers and platforms. A production ML
    // framework will likely implement its own distributions to provide
    // deterministic results.
    auto dist = std::normal_distribution<num_t>{0.0, sigma};

    for (num_t& w : weights_)
    {
        w = dist(rne);
    }

    // NOTE: Setting biases to zero is a common practice, as is initializing the
    // bias to a small value (e.g. on the order of 0.01). It is unclear if the
    // latter produces a consistent result over the former, but the thinking is
    // that a non-zero bias will ensure that the neuron always "fires" at the
    // beginning to produce a signal.
    //
    // Here, we initialize all biases to a small number, but the reader should
    // consider experimenting with other approaches.
    for (num_t& b : biases_)
    {
        b = 0.01;
    }
}

void FFNode::forward(num_t* inputs)
{
    // Remember the last input data for backpropagation later
    last_input_ = inputs;

    for (size_t i = 0; i != output_size_; ++i)
    {
        // For each output vector, compute the dot product of the input data
        // with the weight vector add the bias

        num_t z{0.0};

        size_t offset = i * input_size_;

        for (size_t j = 0; j != input_size_; ++j)
        {
            z += weights_[offset + j] * inputs[j];
        }
        // Add neuron bias
        z += biases_[i];

        switch (activation_)
        {
        case Activation::ReLU:
            activations_[i] = std::max(z, num_t{0.0});
            break;
        case Activation::Softmax:
        default:
            activations_[i] = std::exp(z);
            break;
        }
    }

    if (activation_ == Activation::Softmax)
    {
        // softmax(z)_i = exp(z_i) / \sum_j(exp(z_j))
        num_t sum_exp_z{0.0};
        for (size_t i = 0; i != output_size_; ++i)
        {
            // NOTE: with exploding gradients, it is quite easy for this
            // exponential function to overflow, which will result in NaNs
            // infecting the network.
            sum_exp_z += activations_[i];
        }
        num_t inv_sum_exp_z = num_t{1.0} / sum_exp_z;
        for (size_t i = 0; i != output_size_; ++i)
        {
            activations_[i] *= inv_sum_exp_z;
        }
    }

    // Forward activation data to all subsequent nodes in the computational
    // graph
    for (Node* subsequent : subsequents_)
    {
        subsequent->forward(activations_.data());
    }
}

void FFNode::reverse(num_t* gradients)
{
    // We receive a vector of output_size_ gradients of the loss function with
    // respect to the activations of this node.

    // We need to compute the gradients of the loss function with respect to
    // each parameter in the node (all weights and biases). In addition, we need
    // to compute the gradients with respect to the inputs in order to propagate
    // the gradients further.

    // Notation:
    //
    // Subscripts on any of the following vector and matrix quantities are used
    // to specify a specific element of the vector or matrix.
    //
    //   - I is the input vector
    //   - W is the weight matrix
    //   - B is the bias vector
    //   - Z = W*I + B
    //   - A is our activation function (ReLU or Softmax in this case)
    //   - L is the total loss (cost)
    //
    // The gradient we receive from the subsequent is dJ/dg(Z) which we can use
    // to compute dJ/dW_{i, j}, dJ/dB_i, and dJ/dI_i

    // First, we compute dJ/dz as dJ/dg(z) * dg(z)/dz and store it in our
    // activations array
    for (size_t i = 0; i != output_size_; ++i)
    {
        // dg(z)/dz
        num_t activation_grad{0.0};
        switch (activation_)
        {
        case Activation::ReLU:
            // For a ReLU function, the gradient is unity when the activation
            // exceeds 0.0, and 0.0 otherwise. Technically, the gradient is
            // undefined at 0, but in practice, defining the gradient at this
            // point to be 0 isn't an issue
            if (activations_[i] > num_t{0.0})
            {
                activation_grad = num_t{1.0};
            }
            else
            {
                activation_grad = num_t{0.0};
            }
            // dJ/dz = dJ/dg(z) * dg(z)/dz
            activation_gradients_[i] = gradients[i] * activation_grad;
            break;
        case Activation::Softmax:
        default:
            // F.T.R. The implementation here correctly computes gradients for
            // the general softmax function accounting for all received
            // gradients. However, this step can be optimized significantly if
            // it is known that the softmax output is being compared to a
            // one-hot distribution. The softmax output of a given unit is
            // exp(z_i) / \sum_j exp(z_j). When the loss gradient with respect
            // to the softmax outputs is returned, a single i is selected from
            // among the softmax outputs in a 1-hot encoding, corresponding to
            // the correct classification for this training sample. Complete the
            // derivation for the gradient of the softmax assuming a one-hot
            // distribution and implement the optimized routine.

            for (size_t j = 0; j != output_size_; ++j)
            {
                if (i == j)
                {
                    activation_grad += activations_[i]
                                       * (num_t{1.0} - activations_[i])
                                       * gradients[j];
                }
                else
                {
                    activation_grad
                        += -activations_[i] * activations_[j] * gradients[j];
                }
            }

            activation_gradients_[i] = activation_grad;
            break;
        }
    }

    for (size_t i = 0; i != output_size_; ++i)
    {
        // Next, let's compute the partial dJ/db_i. If we hold all the weights
        // and inputs constant, it's clear that dz/db_i is just 1 (consider
        // differentiating the line mx + b with respect to b). Thus, dJ/db_i =
        // dJ/dg(z_i) * dg(z_i)/dz_i.
        bias_gradients_[i] += activation_gradients_[i];
    }

    // CAREFUL! Unlike the other gradients, we reset input gradients to 0. These
    // values are used primarily as a subexpression in computing upstream
    // gradients and do not participate in the network optimization step (aka
    // Stochastic Gradient Descent) later.
    std::fill(input_gradients_.begin(), input_gradients_.end(), 0);

    // To compute dz/dI_i, recall that z_i = \sum_i W_i*I_i + B_i. That is, the
    // precursor to each activation is a dot-product between a weight vector an
    // the input plus a bias. Thus, dz/dI_i must be the sum of all weights that
    // were scaled by I_i during the forward pass.
    for (size_t i = 0; i != output_size_; ++i)
    {
        size_t offset = i * input_size_;
        for (size_t j = 0; j != input_size_; ++j)
        {
            input_gradients_[j]
                += weights_[offset + j] * activation_gradients_[i];
        }
    }

    for (size_t i = 0; i != input_size_; ++i)
    {
        for (size_t j = 0; j != output_size_; ++j)
        {
            // Each individual weight shows up in the equation for z once and is
            // scaled by the corresponding input. Thus, dJ/dw_i = dJ/dg(z_i) *
            // dg(z_i)/dz_i * dz_i/d_w_ij where the last factor is equal to the
            // input scaled by w_ij.

            weight_gradients_[j * input_size_ + i]
                += last_input_[i] * activation_gradients_[j];
        }
    }

    for (Node* node : antecedents_)
    {
        // Forward loss gradients with respect to the inputs to the previous
        // node.
        //
        // F.T.R. Technically, if the antecedent node has no learnable
        // parameters, there is no point forwarding gradients to that node.
        // Furthermore, if no antecedent nodes required any gradients, we could
        // have skipped computing the gradients for this node altogether. A
        // simple way to implement this is to add a `parameter_count` virtual
        // method on the Node interface leverage it to save some work whenever
        // possible here.
        node->reverse(input_gradients_.data());
    }
}

// F.T.R. It is more efficient to store parameters contiguously so they can be
// accessed without branching or arithmetic.
num_t* FFNode::param(size_t index)
{
    if (index < weights_.size())
    {
        return &weights_[index];
    }
    return &biases_[index - weights_.size()];
}

num_t* FFNode::gradient(size_t index)
{
    if (index < weights_.size())
    {
        return &weight_gradients_[index];
    }
    return &bias_gradients_[index - weights_.size()];
}

void FFNode::print() const
{
    std::printf("%s\n", name_.c_str());

    // Consider the input samples as column vectors, and visualize the weights
    // as a matrix transforming vectors with input_size_ dimension to size_
    // dimension
    std::printf("Weights (%d x %d)\n", output_size_, input_size_);
    for (size_t i = 0; i != output_size_; ++i)
    {
        size_t offset = i * input_size_;
        for (size_t j = 0; j != input_size_; ++j)
        {
            std::printf("\t[%d]%f", offset + j, weights_[offset + j]);
        }
        std::printf("\n");
    }
    std::printf("Biases (%d x 1)\n", output_size_);
    for (size_t i = 0; i != output_size_; ++i)
    {
        std::printf("\t%f\n", biases_[i]);
    }
    std::printf("\n");
}
