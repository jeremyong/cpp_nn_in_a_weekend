add_executable(
    nn
    main.cpp
    CCELossNode.cpp
    FFNode.cpp
    GDOptimizer.cpp
    MNIST.cpp
    Model.cpp
)

target_compile_features(nn PUBLIC cxx_std_17)
