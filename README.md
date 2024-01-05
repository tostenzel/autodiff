![](/edugrad-header.png)

![test workflow badge](https://github.com/tostenzel/edugrad/actions/workflows/Tests.yaml/badge.svg)


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
