import numpy as np
import unittest
from edugrad import Tensor


class TestZeroShapeTensor(unittest.TestCase):
    def test_cat(self):
        s = Tensor.rand(3, 2, 2)
        t = Tensor.rand(3, 2, 0).cat(s, dim=2)
        assert t.shape == (3, 2, 2)
        np.testing.assert_equal(t.numpy(), s.numpy())

        # torch does not support padding non-zero dim with 0-size. torch.nn.functional.pad(torch.zeros(3,2,0), [0,0,0,4,0,0])
        s = Tensor.rand(3, 4, 0)
        t = Tensor.rand(3, 2, 0).cat(s, dim=1)
        assert t.shape == (3, 6, 0)
        np.testing.assert_equal(t.numpy(), np.zeros((3, 6, 0)))
