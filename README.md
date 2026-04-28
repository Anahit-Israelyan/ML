# Machine Learning Foundations and Advanced Architectures

This repository contains a curated collection of implementations, theoretical derivations, and applied modeling techniques across the domains of foundational machine learning, computer vision, and natural language processing. The codebase emphasizes rigorous mathematical foundations, "from-scratch" algorithmic engineering, and state-of-the-art optimization strategies.

## Theoretical Depth

A core focus of this work lies in the fundamental understanding of learning dynamics and optimization landscapes. Theoretical underpinnings are rigorously established prior to implementation. Key derivations include the analytical computation of gradients for arbitrary network topologies and the explicit mathematical formulation of backpropagation. For instance, the gradient of the cross-entropy loss $\mathcal{L}$ with respect to the pre-activation $z^{(L)}$ of the output layer is systematically derived as:

$$
\frac{\partial \mathcal{L}}{\partial z^{(L)}} = \hat{y} - y
$$

These mathematical frameworks form the basis for all custom implementations, ensuring that empirical results are grounded in robust analytical theory.

## From-Scratch Engineering

To demonstrate a profound understanding of neural representations, several architectures and optimization algorithms are engineered entirely from fundamental linear algebra operations using NumPy, intentionally bypassing high-level deep learning abstractions. 

- **Custom Deep Learning Framework**: Implemented a modular neural network framework capable of defining arbitrary sequences of affine transformations and non-linearities. 
- **Forward and Backward Kernels**: Explicit matrix operations are utilized to manage state forward-propagation and error back-propagation. For a given hidden layer $l$, the parameter updates are computed as:

$$
\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})
$$
$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

This low-level engineering ensures an intimate understanding of computational graphs, memory constraints, and numerical stability.

## Advanced Optimization

Beyond foundational frameworks, this repository leverages modern deep learning environments (PyTorch) to explore advanced regularization and data augmentation techniques within deep convolutional neural networks.

- **Custom ResNet Architecture**: Implemented custom ResNet topologies specifically optimized for challenging datasets such as CIFAR-10.
- **Mixup and Cutout Augmentation**: Engineered custom data pipeline transformations to enhance model generalization. For Mixup, virtual training examples $(\tilde{x}, \tilde{y})$ are constructed via convex combinations of randomly selected pairs $(x_i, y_i)$ and $(x_j, y_j)$:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$
$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$. These interventions empirically demonstrate superior regularization properties and robustness to label noise.

## Sequence Modeling

The repository extends into natural language processing, addressing the inherent complexities of sequential data via recurrent topologies.

- **Vocabulary Engineering**: Developed robust, scalable vocabulary classes capable of ingesting raw text, handling tokenization, and executing numerical indexing for large-scale corpora.
- **Recurrent Neural Networks (RNN)**: Architected custom RNN classifiers designed for high-dimensional sentiment analysis. The hidden state transitions are defined rigorously to capture temporal dependencies:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

These implementations highlight capabilities in both the architectural design of sequence models and the preprocessing pipelines critical for unstructured text evaluation.

## Reproducibility

Maintaining rigorous academic and engineering standards, all dependencies required to execute the notebooks and reproduce the training results are documented.

To configure the environment and reproduce the empirical results, execute the following commands within an isolated virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-username/ML.git
cd ML

# Install dependencies
pip install -r requirements.txt
```

Each module contains deterministic seeds where applicable to ensure exact reproducibility of convergence trajectories and evaluation metrics across hardware configurations.
