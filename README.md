# edugrad

## Installation

    git clone https://github.com/tostenzel/edugrad
    cd edugrad

    # Setup environment in edugrad/.env and install requirements with conda from environment.yaml
    conda create --prefix .env
    conda activate .env/
    conda env update --file environment.yaml --prefix .env

    # Install edugrad from source in editable mode to enable absolute imports
    pip install -e .

    # Verify installation
    python applications/learn_mnist.py