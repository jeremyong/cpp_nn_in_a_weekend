#include "Model.hpp"

Node::Node(Model& model, std::string name)
    : model_(model)
    , name_{std::move(name)}
{}

Model::Model(std::string name)
    : name_{std::move(name)}
{}

void Model::create_edge(Node& dst, Node& src)
{
    // NOTE: No validation is done to ensure the edge doesn't already exist
    dst.antecedents_.push_back(&src);
    src.subsequents_.push_back(&dst);
}

rne_t::result_type Model::init(rne_t::result_type seed)
{
    if (seed == 0)
    {
        // Generate a new random seed from the host random device
        std::random_device rd{};
        seed = rd();
    }
    std::printf("Initializing model parameters with seed: %zu\n", seed);

    rne_t rne{seed};

    for (auto& node : nodes_)
    {
        node->init(rne);
    }

    return seed;
}

void Model::print() const
{
    // Invoke "print" on each node in the order added
    for (auto&& node : nodes_)
    {
        node->print();
    }
}
