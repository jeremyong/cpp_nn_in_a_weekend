# Neural Network in a Weekend in C++

This repository is the companion code the article "Neural Network in a Weekend."
Readers are welcome to clone the repository or use the code herein as a reference if following along the article.
Pull requests for errors and bugs are welcome.
However, pull requests that introduce new features are unlikely to be considered, as the ultimate goal of this code is to be tractable for a newer practitioner getting started with deep learning architecutres.

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
Data is stored with the MSB first, meaning that on a little-endian architecture, the bytes must be flipped.
