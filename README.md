# C++ Neural Network in a Weekend

This repository is the companion code the article "Neural Network in a Weekend."
Readers are welcome to clone the repository or use the code herein as a reference if following along the article.
Pull requests and issues filed for errors and bugs in both code and/or documentation are welcome and appreciated.
However, pull requests that introduce new features are unlikely to be considered, as the ultimate goal of this code is to be tractable for a newer practitioner getting started with deep learning architectures.

## Compilation and Usage

    mkdir build
    cd build
    # substitute Ninja for your preferred generator
    cmake .. -G Ninja
    ninja
    # trains the network and writes the learned parameters to disk
    ./src/nn train ../data/train
    # evaluate the model loss and accuracy based on the trained parameters
    ./src/nn evaluate ../data/test ./ff.params

## Conventions

1. Member variables have a single underscore suffix (e.g. `member_variable_`)
2. The `F.T.R.` acroynym stands for "For the reader" and precedes suggestions for experimentation, improvements, or alternative implementations
3. Throughout, you may see the type aliases `num_t` and `rne_t`.
   These aliases refer to `float` and `std::mt199837` respectively and are defined in `Model.hpp` to easily experiment with alternative precisions and random number engines.
   The reader may wish to make these parameters changeable by other means.

## General Code Structure

The neural network is modeled as a computational graph. The graph itself is the `Model` defined in `Model.hpp`.
Nodes in the computational graph override the `Node` base class and must implement various methods to explain how data flows through the node (forwards and backwards).

The fully-connected feedforward node in this example is implemented as `FFNode` in `FFNode.hpp`.
The cross-entropy loss node is implemented in `CELossNode.hpp`.
Together, these two nodes are all that is needed to train our example on the MNIST dataset.

## Data

For your convenient, the MNIST data used to train and test the network is provided uncompressed in the `data/` subdirectory.
The data is structured like so:

### Images

Image data can be parsed using code provided in the `MNIST.hpp` header, but the data is described here as well.
Multi-byte integers are stored with the MSB first, meaning that on a little-endian architecture, the bytes must be flipped.
Image pixel data is stored in row-major order and packed contiguously one after another.

     Bytes
    [00-03] 0x00000803 (Magic Number: 2051)
    [04-07] image count
    [08-11] rows
    [12-15] columns
    [16]    pixel[0, 0]
    [17]    pixel[0, 1]
    ...

### Labels

Label data is parsed according to the following byte layout:

     Bytes
    [00-03] 0x00000801 (Magic Number: 2049)
    [04-07] label count
    [8]     label 1
    [9]     label 2
    ...

The parser provided by the `MNIST` input node validates the magic numbers to ensure the machine endianness is as expected, and also validates that the image data and label data sizes match.
