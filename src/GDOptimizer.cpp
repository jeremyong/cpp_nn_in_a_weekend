#include "GDOptimizer.hpp"
#include "Model.hpp"
#include <cmath>

GDOptimizer::GDOptimizer(num_t eta)
    : eta_{eta}
{}

void GDOptimizer::train(Node& node)
{
    size_t param_count = node.param_count();
    // std::printf("%s Param count: %zu\n", node.name().c_str(), param_count);
    for (size_t i = 0; i != param_count; ++i)
    {
        num_t& param    = *node.param(i);
        num_t& gradient = *node.gradient(i);

        param = param - eta_ * gradient;

        // Reset the gradient which will be accumulated again in the next
        // training epoch
        gradient = num_t{0.0};
    }
}
