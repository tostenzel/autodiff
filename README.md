![](/edugrad-header.png)


**edugrad** is the most simple and minimalistic implementation of a deep learning framework. Its purpose is to demystify the core components of such libraries, explaining their functionality and underlying programming principles.

## Key Features
- **Autograd Mechanism**: Features an automatic differentiation system for computing gradients, essential for training neural networks.
- **Tensor Operations**: Implements fundamental tensor/matrix operations crucial for neural network computations in numpy.
- **Simple Interface**: Offers an API that mirrors PyTorch
- **Educational Code**: The code style and module structure are designed for ease of understanding, both programmatically and conceptually.

Please note that while edugrad theoretically supports the implementation of any neural network model, it lacks the memory and computational optimizations found in more advanced frameworks. This design choice maximizes code readability but limits the framework's capability to smaller models.

![test workflow badge](https://github.com/tostenzel/edugrad/actions/workflows/Tests.yaml/badge.svg)


## Code Structure

## Credits

## Conceptual Explanations


## Installation

```
git clone https://github.com/tostenzel/edugrad
cd edugrad
```

Set up environment in edugrad/.env and install requirements with conda from environment.yaml:
```
conda create --prefix .env
conda activate .env/
conda env update --file environment.yaml --prefix .env
```
Install edugrad from source in editable mode to enable absolute imports:
```
pip install -e .
```
Verify installation:
```
python applications/learn_mnist.py
```
