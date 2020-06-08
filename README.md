This repository contains impemenations of Variational Autoencoders(VAE) and various Generative Adversarial Network(GAN).

## Variational Autoencoders:
Variational Autoencoders (VAE) model trained on MNIST and CIFAR10 datasets. The two key hyper-parameters, latent space dimension as well as model complexity were investigated. In the context of VAE, the model complexity is defined as the intermediate dimension representing the dimension of the dense layer receiving the input.

## Deep Convolution Generative Adversarial Network (DCGAN)
Generative Adversarial Network (GAN) model trained on MNIST and CIFAR10 datasets. The two key hyper-parameters,
latent space dimension as well as model complexity were investigated. The model complexity is defined as the intermediate dimension representing the dimension of the dense layers receiving the input. As vanilla GANs are rather unstable, an alternate is the Deep Convolution GAN (DCGAN) model which contain features like convolutional layers and batch normalization which can help with the stability of the convergence.

## Wasserstein Generative Adversarial Network(WGAN)
The goal of WGAN is to better stabilize GAN training and diminish some disadvantages of other GAN models such as mode collapse and uninformative loss. Thus, WGAN was implemented with properties such as a meaningful loss metric to correlate generator's convergence and quality of amples. Also, it can improve the overall stability of the optimization process.


