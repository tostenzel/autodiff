import sys
print(sys.path)

import pytest
from applications.learn_mnist import train_and_evaluate_mnist

def test_mnist_accuracy():
    test_accuracy = train_and_evaluate_mnist()
    assert test_accuracy > 0.93, f"Test accuracy too low: {test_accuracy}"