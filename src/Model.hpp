#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Default precision: single
using num_t = float;
// Default random number engine: 32-bit Mersenne Twister by Matsumoto and
// Nishimura, 1998
using rne_t = std::mt19937;

enum class Activation
{
    ReLU,
    Softmax
};

class Model;

// Base class of computational nodes in a model
class Node
{
public:
    Node(Model& model, std::string name);

    // Initialize the parameters of the node with a provided random number
    // engine.
    virtual void init(rne_t& rne) = 0;

    // Data is fed forward through the network using a simple generic interface.
    // We do this to avoid requiring an involved N-dimensional matrix
    // abstraction. Here, the "shape" of the data is dependent on the Node's
    // implementation and the way a given Node is initialized.
    //
    // In practice, this should be replaced with an actual type with a shape
    // defined by data to permit additional validation. It is also common for
    // the data object passed here to not contain the data directly (the data
    // may be located on a GPU for example)
    virtual void forward(num_t* inputs) = 0;

    // Expected inputs during the reverse accumulation phase are the loss
    // gradients with respect to each output
    //
    // The node is expected to compute the loss gradient with respect to each
    // parameter and update the parameter according to the model's optimizer,
    // after which, the gradients with respect to the node inputs are propagated
    // backwards again.
    virtual void reverse(num_t* gradients) = 0;

    // Returns the number of learnable parameters in this node. Nodes that are
    // input or loss nodes have no learnable parameters.
    virtual size_t param_count() const noexcept
    {
        return 0;
    }

    // Indexing operator for learnable parameters that are mutated during
    // training. Nodes without learnable parameters should keep this
    // unimplemented.
    virtual num_t* param(size_t index)
    {
        return nullptr;
    }

    // Indexing operator for the loss gradient with respect to a learnable
    // parameter. Used by an optimizer to adjust the corresponding parameter and
    // potentially for tracking gradient histories (done in more sophisticated
    // optimizers, e.g. AdaGrad)
    virtual num_t* gradient(size_t index)
    {
        return nullptr;
    }

    [[nodiscard]] std::string const& name() const noexcept
    {
        return name_;
    }

    // Generic function that displays the contents of the node in some fashion
    virtual void print() const = 0;

protected:
    friend class Model;

    Model& model_;
    std::string name_;
    std::vector<Node*> antecedents_;
    std::vector<Node*> subsequents_;
};

// Base class of optimizer used to train a model
class Optimizer
{
public:
    virtual void train(Node& node) = 0;
};

class Model
{
public:
    Model(std::string name);

    template <typename Node_t, typename... T>
    Node_t& add_node(T&&... args)
    {
        nodes_.emplace_back(
            std::make_unique<Node_t>(*this, std::forward<T>(args)...));
        return reinterpret_cast<Node_t&>(*nodes_.back());
    }

    void create_edge(Node& dst, Node& src);

    // Initialize the parameters of all nodes with the provided seed. If the
    // seed is 0, a new random seed is chosen instead. Returns the seed used.
    rne_t::result_type init(rne_t::result_type seed = 0);

    void print() const;

private:
    friend class Node;

    std::string name_;
    std::vector<std::unique_ptr<Node>> nodes_;
};
