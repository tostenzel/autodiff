"""Train a classifier to recognize the hand-written digit on gray-scale images.

and evaluate the results.

"""

import os
import gzip

import numpy as np

from edugrad import Tensor
from edugrad.optim import optimizer


def fetch_mnist(for_convolution=True):
    def parse(file):
        return np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()

    # parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    BASE = os.path.dirname(__file__) + "/datasets"

    X_train = parse(BASE + "/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28 * 28)).astype(np.float32)
    Y_train = parse(BASE + "/mnist/train-labels-idx1-ubyte.gz")[8:].astype(np.int32)
    X_test = parse(BASE + "/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28 * 28)).astype(np.float32)
    Y_test = parse(BASE + "/mnist/t10k-labels-idx1-ubyte.gz")[8:].astype(np.int32)
    if for_convolution:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
    return X_train, Y_train, X_test, Y_test


class ConvNet:
    def __init__(self):
        # https://keras.io/examples/vision/mnist_convnet/
        kernel_sz = 3
        in_chan, out_chan = 8, 16  # Reduced from 32, 64 -> Faster training
        self.c1 = Tensor.scaled_uniform(in_chan, 1, kernel_sz, kernel_sz)
        self.c2 = Tensor.scaled_uniform(out_chan, in_chan, kernel_sz, kernel_sz)
        self.l1 = Tensor.scaled_uniform(out_chan * 5 * 5, 10)

    def __call__(self, x: Tensor):
        x = x.conv2d(self.c1).relu().max_pool2d()
        x = x.conv2d(self.c2).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        return x.dot(self.l1).log_softmax()


def train_and_evaluate_mnist(num_steps=100, batch_size=128, learning_rate=0.001):
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    model = ConvNet()
    opt = optimizer.Adam([model.c1, model.c2, model.l1], lr=learning_rate)

    with Tensor.train():
        for step in range(num_steps):
            samp = np.random.randint(0, X_train.shape[0], size=(batch_size))
            xb, yb = Tensor(X_train[samp], requires_grad=False), Tensor(Y_train[samp])

            out = model(xb)
            loss = out.sparse_categorical_crossentropy(yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Evaluate Train
            y_preds = out.numpy().argmax(axis=-1)
            acc = (y_preds == yb.numpy()).mean()
            if step == 0 or (step + 1) % 20 == 0:
                print(f"Step {step+1:<3} | Loss: {loss.numpy():.4f} | Train Acc: {acc:.3f}")

    # Evaluate Test
    test_accuracy = 0
    for i in range(0, len(Y_test), batch_size):
        xb, yb = Tensor(X_test[i : i + batch_size], requires_grad=False), Tensor(Y_test[i : i + batch_size])
        out = model(xb)
        preds = out.argmax(axis=-1)
        test_accuracy += (preds == yb).sum().numpy()
    test_accuracy /= len(Y_test)
    return test_accuracy


if __name__ == "__main__":
    # Only execute if this script is run directly
    test_accuracy = train_and_evaluate_mnist()
    print(f"Test Acc: {test_accuracy:.3f}")
