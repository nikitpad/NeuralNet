# NeuralNet

A library for creating and training neural networks for .NET programs.

## Features:
- Fully connected (dense) layers
- Pooling layers
- Convolution layers
- Stochastic gradient descent training with Adam
- Ability to create own layers
- Several activation functions (ReLU, Leaky ReLU, Sigmoid)
- Support for serializing network parameters into JSON data (other formats may be added in the future)

It's unoptimized at the moment and supports only CPU training, but multi-layer perceptrons (MLPs) and CNNs have been shown to successfully train and classify handwritten digits at relatively high accuracy.
[A CNN being evaluated on the MNIST test data examples](https://i.imgur.com/PIFc9Gl.png)