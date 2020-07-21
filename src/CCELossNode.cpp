#include "CCELossNode.hpp"
#include <limits>

CCELossNode::CCELossNode(Model& model,
                         std::string name,
                         uint16_t input_size,
                         size_t batch_size)
    : Node{model, std::move(name)}
    , input_size_{input_size}
    , inv_batch_size_{num_t{1.0} / static_cast<num_t>(batch_size)}
{
    // When we deliver a gradient back, we deliver just the loss gradient with
    // respect to any input and the index that was "hot" in the second argument.
    gradients_.resize(input_size_);
}

void CCELossNode::forward(num_t* data)
{
    // The cross-entropy categorical loss is defined as -\sum_i(q_i * log(p_i))
    // where p_i is the predicted probabilty and q_i is the expected probablity
    //
    // In information theory, by convention, lim_{x approaches 0}(x log(x)) = 0

    num_t max{0.0};
    size_t max_index;

    loss_ = num_t{0.0};
    for (size_t i = 0; i != input_size_; ++i)
    {
        if (data[i] > max)
        {
            max_index = i;
            max       = data[i];
        }

        // Because the target vector is one-hot encoded, most of these terms
        // will be zero, but we leave the full calculation here to be explicit
        // and in the event we want to compute losses against probability
        // distributions that arent one-hot. In practice, a faster code path
        // should be employed if the targets are known to be one-hot
        // distributions.
        loss_ -= target_[i]
                 * std::log(
                     // Prevent undefined results when taking the log of 0
                     std::max(data[i], std::numeric_limits<num_t>::epsilon()));

        if (target_[i] != num_t{0.0})
        {
            active_ = i;
        }

        // NOTE: The astute reader may notice that the gradients associated with
        // many of the loss node's input signals will be zero because the
        // cross-entropy is performed with respect to a one-hot vector.
        // Fortunately, because the layer preceding the output layer is a
        // softmax layer, the gradient from the single term contributing in the
        // above expression has a dependency on *every* softmax output unit (all
        // outputs show up in the summation in the softmax denominator).
    }

    if (max_index == active_)
    {
        ++correct_;
    }
    else
    {
        ++incorrect_;
    }

    cumulative_loss_ += loss_;

    // Store the data pointer to compute gradients later
    last_input_ = data;
}

void CCELossNode::reverse(num_t* data)
{
    // dL/dq_i = d(-\sum_i(p_i log(q_i)))/dq_i = -1 / q_j where j is the index
    // of the correct classification (loss gradient for a single sample).
    //
    // Note the normalization factor where we multiply by the inverse batch
    // size. This ensures that losses computed by the network are similar in
    // scale irrespective of batch size.

    for (size_t i = 0; i != input_size_; ++i)
    {
        gradients_[i] = -inv_batch_size_ * target_[i] / last_input_[i];
    }

    for (Node* node : antecedents_)
    {
        node->reverse(gradients_.data());
    }
}

void CCELossNode::print() const
{
    std::printf("Avg Loss: %f\t%f%% correct\n", avg_loss(), accuracy());
}

num_t CCELossNode::accuracy() const
{
    return static_cast<num_t>(correct_)
           / static_cast<num_t>(correct_ + incorrect_);
}
num_t CCELossNode::avg_loss() const
{
    return cumulative_loss_ / static_cast<num_t>(correct_ + incorrect_);
}

void CCELossNode::reset_score()
{
    cumulative_loss_ = num_t{0.0};
    correct_         = 0;
    incorrect_       = 0;
}
