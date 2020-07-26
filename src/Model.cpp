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
    std::printf("Initializing model parameters with seed: %u\n", seed);

    rne_t rne{seed};

    for (auto& node : nodes_)
    {
        node->init(rne);
    }

    return seed;
}

void Model::train(Optimizer& optimizer)
{
    for (auto&& node : nodes_)
    {
        optimizer.train(*node);
    }
}

void Model::print() const
{
    // Invoke "print" on each node in the order added
    for (auto&& node : nodes_)
    {
        node->print();
    }
}

void Model::save(std::ofstream& out)
{
    // To save the model to disk, we employ a very simple scheme. All nodes are
    // looped through in the order they were added to the model. Then, all
    // advertised learnable parameters are serialized in host byte-order to the
    // supplied output stream.
    //
    // F.T.R. This simplistic method of saving the model to disk isn't very
    // robust or practical in the real world. For one thing, it contains no
    // reflection data about the topology of the model. Loading the data relies
    // on the model being constructed in the same manner it was trained on.
    // Furthermore, the data will be parsed incorrectly if the program is
    // recompiled to operate with a different precision. Adopting a more
    // sensible serialization scheme is left as an exercise.
    for (auto& node : nodes_)
    {
        size_t param_count = node->param_count();
        for (size_t i = 0; i != param_count; ++i)
        {
            out.write(
                reinterpret_cast<char const*>(node->param(i)), sizeof(num_t));
        }
    }
}

void Model::load(std::ifstream& in)
{
    for (auto& node : nodes_)
    {
        size_t param_count = node->param_count();
        for (size_t i = 0; i != param_count; ++i)
        {
            in.read(reinterpret_cast<char*>(node->param(i)), sizeof(num_t));
        }
    }
}
