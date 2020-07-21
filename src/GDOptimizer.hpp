#pragma once

#include "Model.hpp"

// Note that this class defines the general gradient descent algorithm. It can
// be used as part of the *Stochastic* gradient descent algorithm (aka SGD) by
// invoking it after smaller batches of training data are evaluated.
class GDOptimizer : public Optimizer
{
public:
    // "Eta" is the commonly accepted character used to denote the learning
    // rate. Given a loss gradient dL/dp for some parameter p, during gradient
    // descent, p will be adjusted such that p' = p - eta * dL/dp.
    GDOptimizer(num_t eta);

    // This should be invoked at the end of each batch's evaluation. The
    // interface technically permits the use of different optimizers for
    // different segments of the computational graph.
    void train(Node& node) override;

private:
    num_t eta_;
};
