# Deep Lossless Compression: PixelCNN + ANS

This project implements a lossless image compression system combining deep autoregressive modeling with modern entropy coding. By learning the probability distribution of images using **PixelCNN** and encoding them via **Asymmetric Numeral Systems (ANS)**, this model achieves high-efficiency compression on the CIFAR-10 domain.

## Probabilistic Model: PixelCNN
The core of the compression engine is **PixelCNN**, a deep autoregressive network. Unlike standard autoencoders, PixelCNN explicitly models the joint distribution of pixels $p(x)$ as a product of conditional distributions:

$$p(x) = \prod_{i=1}^{n^2} p(x_i | x_1, \dots, x_{i-1})$$



* **Masked Convolutions:** It uses strictly masked convolutional filters to ensure that the prediction for pixel $x_i$ depends *only* on previously seen pixels (above and to the left), preserving the autoregressive property.
* **Discrete Distribution:** The model outputs a softmax distribution over pixel values (0–255) for each color channel.

## Training Data: CIFAR-10
The model is trained on the **CIFAR-10** dataset to learn the specific statistical priors of natural object images.
* **Input:** $32 \times 32$ RGB images.
* **Objective:** The network minimizes the Negative Log-Likelihood (NLL) of the training data, effectively learning to predict the "next pixel" with high confidence. Lower NLL directly correlates to lower theoretical bits-per-dimension (bpd).

## Entropy Coding: Asymmetric Numeral Systems (ANS)
To convert the probabilities predicted by PixelCNN into a compressed bitstream, we utilize **rANS (range Asymmetric Numeral Systems)**.

* **The Workflow:**
    1.  **Inference:** PixelCNN processes the image pixel-by-pixel (or in groups).
    2.  **Probability Map:** For every pixel, the network outputs a probability distribution estimate.
    3.  **Encoding:** The ANS encoder uses these specific probabilities to pack the actual pixel value into the bitstream.

### System Summary
| Component | Function |
| :--- | :--- |
| **PixelCNN** | Predicts the probability of pixel $x_i$ based on context. |
| **CIFAR-10** | Provides the prior knowledge (weights) for the model. |
| **ANS** | Losslessly maps pixels to bits using PixelCNN's predictions. |
