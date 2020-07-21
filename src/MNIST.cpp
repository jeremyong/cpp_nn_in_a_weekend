#include "MNIST.hpp"

#include <cstdio>
#include <stdexcept>

// Read sizeof(T) bytes and reverse them to return an unsigned integer on LE
// architectures
template <typename T>
T read_be(std::ifstream& in);

template <>
uint32_t read_be(std::ifstream& in)
{
    char buf[4];
    in.read(buf, 4);

    std::swap(buf[0], buf[3]);
    std::swap(buf[1], buf[2]);
    return *reinterpret_cast<uint32_t*>(buf);
}

MNIST::MNIST(Model& model, std::ifstream& images, std::ifstream& labels)
    : Node{model, "MNIST input"}
    , images_{images}
    , labels_{labels}
{
    // Confirm that passed input file streams are well-formed MNIST data sets
    uint32_t image_magic = read_be<uint32_t>(images);
    if (image_magic != 2051)
    {
        throw std::runtime_error{"Images file appears to be malformed"};
    }
    image_count_ = read_be<uint32_t>(images);

    uint32_t labels_magic = read_be<uint32_t>(labels);
    if (labels_magic != 2049)
    {
        throw std::runtime_error{"Images file appears to be malformed"};
    }

    uint32_t label_count = read_be<uint32_t>(labels);
    if (label_count != image_count_)
    {
        throw std::runtime_error(
            "Label count did not match the number of images supplied");
    }

    uint32_t rows    = read_be<uint32_t>(images);
    uint32_t columns = read_be<uint32_t>(images);
    if (rows != 28 || columns != 28)
    {
        throw std::runtime_error{
            "Expected 28x28 images, non-MNIST data supplied"};
    }

    printf("Loaded images file with %d entries\n", image_count_);
}

void MNIST::forward(num_t* data)
{
    read_next();
    data_[0] = 0.3;
    data_[1] = 0.7;
    for (Node* node : subsequents_)
    {
        node->forward(data_);
    }
}

void MNIST::print() const
{
    // No learned parameters to display for an MNIST input node
}

void MNIST::read_next()
{
    images_.read(buf_, DIM);
    num_t inv = num_t{1.0} / num_t{255.0};
    for (size_t i = 0; i != DIM; ++i)
    {
        data_[i] = static_cast<uint8_t>(buf_[i]) * inv;
    }

    char label;
    labels_.read(&label, 1);

    for (size_t i = 0; i != 10; ++i)
    {
        label_[i] = num_t{0.0};
    }
    label_[static_cast<uint8_t>(label)] = num_t{1.0};
}

void MNIST::print_last()
{
    for (size_t i = 0; i != 10; ++i)
    {
        if (label_[i] == num_t{1.0})
        {
            printf("This is a %zu:\n", i);
            break;
        }
    }

    for (size_t i = 0; i != 28; ++i)
    {
        size_t offset = i * 28;
        for (size_t j = 0; j != 28; ++j)
        {
            if (data_[offset + j] > num_t{0.5})
            {
                if (data_[offset + j] > num_t{0.9})
                {
                    printf("#");
                }
                else if (data_[offset + j] > num_t{0.7})
                {
                    printf("*");
                }
                else
                {
                    printf(".");
                }
            }
            else
            {
                printf(" ");
            }
        }
        printf("\n");
    }
    printf("\n");
}
