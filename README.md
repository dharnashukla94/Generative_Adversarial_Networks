# Variational Autoencoders:
This repository contains implementation of the Variational Autoencoders (VAE) model trained on MNIST and CIFAR10 datasets.
The two key hyper-parameters, latent space dimension as well as model complexity were investigated. In the context of VAE, the
model complexity is defined as the intermediate dimension representing the dimension of the dense layer receiving the input.

# Generative Adversarial Network
## DCGAN
Generative Adversarial Network (GAN) model trained on MNIST and CIFAR10 datasets. The two key hyper-parameters,
latent space dimension as well as model complexity were investigated. The model complexity is defined as the intermediate dimension representing the dimension of the dense layers receiving the input. As vanilla GANs are rather unstable, an alternate is the Deep Convolution GAN (DCGAN) model which contain features like convolutional layers and batch normalization which can help with the stability of the convergence.
