#include "CCELossNode.hpp"
#include "FFNode.hpp"
#include "GDOptimizer.hpp"
#include "MNIST.hpp"
#include "Model.hpp"
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <filesystem>

void train(char* argv[])
{
    // Uncomment to debug floating point instability in the network
    // feenableexcept(FE_INVALID | FE_OVERFLOW);

    std::printf("Executing training routine\n");

    std::ifstream images{
        std::filesystem::path{argv[0]} / "train-images-idx3-ubyte",
        std::ios::binary};

    std::ifstream labels{
        std::filesystem::path{argv[0]} / "train-labels-idx1-ubyte",
        std::ios::binary};

    // Here we create a simple fully-connected feedforward neural network
    Model model{"ff"};
    size_t batch_size = 100;

    MNIST& mnist = model.add_node<MNIST>(images, labels);

    FFNode& hidden = model.add_node<FFNode>("hidden", Activation::ReLU, 32, 784);

    FFNode& output
        = model.add_node<FFNode>("output", Activation::Softmax, 10, 32);

    CCELossNode& loss = model.add_node<CCELossNode>("loss", 10, batch_size);
    loss.set_target(mnist.label());

    // F.T.R. The structure of our computational graph is completely sequential.
    // In fact, the fully connected node and loss node we've implemented here do
    // not support multiple inputs. Consider adding nodes that support "skip"
    // connections that forward outputs from earlier nodes to downstream nodes
    // that aren't directly adjacent (such skip nodes are used in the ResNet
    // architecture)
    model.create_edge(hidden, mnist);
    model.create_edge(output, hidden);
    model.create_edge(loss, output);

    model.init();

    // The gradient descent optimizer is stateless, but other optimizers may not
    // be. Some optimizers need to track "momentum" or gradient histories.
    // Others may slow the learning rate for each parameter at different rates
    // depending on various factors.
    //
    // F.T.R. Implement an alternative SGDOptimizer that decays the learning
    // rate over time and compare the results against this optimizer that learns
    // at a fixed rate.
    GDOptimizer optimizer{num_t{0.3}};

    for (size_t i = 0; i != 400; ++i)
    {
        for (size_t j = 0; j != batch_size; ++j)
        {
            mnist.forward();
            loss.reverse();
        }

        optimizer.train(output);
        optimizer.train(hidden);
        loss.print();
        loss.reset_score();
    }
}

void evaluate(char* argv[])
{
    std::printf("Executing evaluation routine\n");
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::printf("Supported commands include:\ntrain\nevaluate\n");
        return 1;
    }

    if (strcmp(argv[1], "train") == 0)
    {
        train(argv + 2);
    }
    else if (strcmp(argv[1], "evaluate") == 0)
    {
        evaluate(argv + 2);
    }
    else
    {
        std::printf("Argument %s is an unrecognized directive.\n", argv[1]);
    }

    return 0;
}
