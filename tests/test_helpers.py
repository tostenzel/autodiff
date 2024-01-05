import unittest
from edugrad import Tensor


class TestEdugrad(unittest.TestCase):
    def test_argfix(self):
        self.assertEqual(Tensor.zeros().shape, ())
        self.assertEqual(Tensor.ones().shape, ())

        self.assertEqual(Tensor.zeros([]).shape, ())
        self.assertEqual(Tensor.ones([]).shape, ())

        self.assertEqual(Tensor.zeros(tuple()).shape, ())
        self.assertEqual(Tensor.ones(tuple()).shape, ())

        self.assertEqual(Tensor.zeros(1).shape, (1,))
        self.assertEqual(Tensor.ones(1).shape, (1,))

        self.assertEqual(Tensor.zeros(1, 10, 20).shape, (1, 10, 20))
        self.assertEqual(Tensor.ones(1, 10, 20).shape, (1, 10, 20))

        self.assertEqual(Tensor.zeros([1]).shape, (1,))
        self.assertEqual(Tensor.ones([1]).shape, (1,))

        self.assertEqual(Tensor.zeros([10, 20, 40]).shape, (10, 20, 40))
        self.assertEqual(Tensor.ones([10, 20, 40]).shape, (10, 20, 40))

        self.assertEqual(Tensor.rand(1, 10, 20).shape, (1, 10, 20))
        self.assertEqual(Tensor.rand((10, 20, 40)).shape, (10, 20, 40))

        self.assertEqual(Tensor.empty(1, 10, 20).shape, (1, 10, 20))
        self.assertEqual(Tensor.empty((10, 20, 40)).shape, (10, 20, 40))
