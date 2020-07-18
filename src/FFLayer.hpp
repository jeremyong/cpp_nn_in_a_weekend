#pragma once

#include "Dual.hpp"

#include <cstdint>
#include <vector>

// Fully-connected, feedforward Layer

// A feedforward layer is parameterized by the number of neurons it posesses and
// the number of neurons in the layer preceding it
class FFLayer
{
public:
private:
    uint16_t size_;
    uint16_t input_size_;

    // weights_.size() := size_ * input_size_
    std::vector<Dual> weights_;
    // biases_.size() := input_size_
    std::vector<Dual> biases_;
};