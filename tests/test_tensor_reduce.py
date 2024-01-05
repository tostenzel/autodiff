import numpy as np
import unittest, copy
from edugrad import Tensor
from edugrad.dtypes import dtypes


class TestZeroShapeTensor(unittest.TestCase):
    def test_reduce_over_non_zero(self):
        a = Tensor.ones(3, 2, 0).sum(axis=1)
        assert a.shape == (3, 0)
        np.testing.assert_equal(a.numpy(), np.sum(np.zeros((3, 2, 0)), axis=1))

    def test_reduce_over_zero(self):
        a = Tensor.ones(3, 2, 0).sum(axis=2)
        assert a.shape == (3, 2)
        np.testing.assert_equal(a.numpy(), np.sum(np.zeros((3, 2, 0)), axis=2))

        a = Tensor.ones(3, 2, 0).sum(axis=2, keepdim=True)
        assert a.shape == (3, 2, 1)
        np.testing.assert_equal(a.numpy(), np.sum(np.zeros((3, 2, 0)), axis=2, keepdims=True))

    def test_reduce_default(self):
        np.testing.assert_equal(Tensor([]).max().numpy(), -float("inf"))
        np.testing.assert_equal(Tensor([]).min().numpy(), float("inf"))
        np.testing.assert_equal(Tensor([]).sum().numpy(), 0)
        np.testing.assert_equal(Tensor([]).mean().numpy(), 0)
