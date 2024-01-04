import numpy as np
import unittest
from edugrad import Tensor


class TestZeroShapeTensor(unittest.TestCase):
    def test_reshape(self):
        t = Tensor.zeros(3, 2, 0)
        a = t.reshape(7, 0)
        assert a.shape == (7, 0)
        np.testing.assert_equal(a.numpy(), np.zeros((7, 0)))
        with self.assertRaises(ValueError):
            # cannot reshape array of size 0 into shape ()
            a = t.reshape(())

    def test_expand(self):
        t = Tensor.full((3, 2, 0), 12)
        # with numpy operands could not be broadcast together with remapped shapes [original->remapped]: (3,2,0)
        # and requested shape (6,2,0)
        with self.assertRaises(ValueError):
            t = t.expand((6, 2, 0))
            # assert t.shape == (6, 2, 0)
            # np.testing.assert_equal(t.numpy(), np.full((6, 2, 0), 12))

    def test_pad(self):
        t = Tensor.rand(3, 2, 0).pad((None, None, (1, 1)), 1)
        assert t.shape == (3, 2, 2)
        np.testing.assert_equal(t.numpy(), np.ones((3, 2, 2)))

        # torch does not support padding non-zero dim with 0-size. torch.nn.functional.pad(torch.zeros(3,2,0), [0,0,0,4,0,0])
        t = Tensor.rand(3, 2, 0).pad((None, (1, 1), None), 1)
        assert t.shape == (3, 4, 0)
        np.testing.assert_equal(t.numpy(), np.ones((3, 4, 0)))

        t = Tensor.rand(3, 2, 0).pad(((1, 1), None, None), 1)
        assert t.shape == (5, 2, 0)
        np.testing.assert_equal(t.numpy(), np.ones((5, 2, 0)))

    def test_shrink_into_zero(self):
        t = Tensor.rand(3, 4)
        assert t.shrink((None, (2, 2))).shape == (3, 0)
        assert t.shrink(((2, 2), None)).shape == (0, 4)
        assert t.shrink(((2, 2), (2, 2))).shape == (0, 0)
