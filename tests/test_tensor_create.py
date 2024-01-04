import numpy as np
import unittest
from edugrad import Tensor
from edugrad.dtypes import dtypes


class TestTinygrad(unittest.TestCase):
    def test_zerodim_initialization(self):
        a = Tensor(55)
        b = Tensor(3.14)

        self.assertEqual(a.shape, ())
        self.assertEqual(b.shape, ())

    def test_plus_equals(self):
        a = Tensor.randn(10, 10)
        b = Tensor.randn(10, 10)
        c = a + b
        val1 = c.numpy()
        a += b
        val2 = a.numpy()
        np.testing.assert_allclose(val1, val2)

    def test_random_fns_are_deterministic_with_seed(self):
        for random_fn in [Tensor.randn, Tensor.normal, Tensor.uniform, Tensor.scaled_uniform]:
            with self.subTest(msg=f"Tensor.{random_fn.__name__}"):
                Tensor.manual_seed(1337)
                a = random_fn(10, 10)
                Tensor.manual_seed(1337)
                b = random_fn(10, 10)
                np.testing.assert_allclose(a.numpy(), b.numpy())

    def test_randn_isnt_inf_on_zero(self):
        # simulate failure case of rand handing a zero to randn
        original_rand, Tensor.rand = Tensor.rand, Tensor.zeros
        try:
            self.assertNotIn(np.inf, Tensor.randn(16).numpy())
        except:
            raise
        finally:
            Tensor.rand = original_rand

    def test_zeros_like_has_same_dtype_and_shape(self):
        for datatype in [dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8]:
            a = Tensor([1, 2, 3], dtype=datatype)
            b = Tensor.zeros_like(a)
            assert a.dtype == b.dtype, f"dtype mismatch {a.dtype=} != {b.dtype}"
            assert a.shape == b.shape, f"shape mismatch {a.shape} != {b.shape}"

        a = Tensor([1, 2, 3])
        b = Tensor.zeros_like(a, dtype=dtypes.int8)
        assert (
            a.dtype == dtypes.only_int and b.dtype == dtypes.int8
        ), "a.dtype should be int and b.dtype should be char"
        assert a.shape == b.shape, f"shape mismatch {a.shape} != {b.shape}"

    def test_ones_like_has_same_dtype_and_shape(self):
        for datatype in [dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8]:
            a = Tensor([1, 2, 3], dtype=datatype)
            b = Tensor.ones_like(a)
            assert a.dtype == b.dtype, f"dtype mismatch {a.dtype=} != {b.dtype}"
            assert a.shape == b.shape, f"shape mismatch {a.shape} != {b.shape}"

        a = Tensor([1, 2, 3])
        b = Tensor.ones_like(a, dtype=dtypes.int8)
        assert (
            a.dtype == dtypes.only_int and b.dtype == dtypes.int8
        ), "a.dtype should be int and b.dtype should be char"
        assert a.shape == b.shape, f"shape mismatch {a.shape} != {b.shape}"


class TestZeroShapeTensor(unittest.TestCase):
    def test_rand(self):
        t = Tensor.rand(3, 2, 0)
        assert t.shape == (3, 2, 0)
        np.testing.assert_equal(t.numpy(), np.zeros((3, 2, 0)))
        t = Tensor.rand(0)
        assert t.shape == (0,)
        np.testing.assert_equal(t.numpy(), np.zeros((0,)))
        t = Tensor.rand(0, 0, 0)
        assert t.shape == (0, 0, 0)
        np.testing.assert_equal(t.numpy(), np.zeros((0, 0, 0)))

    def test_full(self):
        t = Tensor.zeros(3, 2, 0)
        assert t.shape == (3, 2, 0)
        np.testing.assert_equal(t.numpy(), np.zeros((3, 2, 0)))
        t = Tensor.full((3, 2, 0), 12)
        assert t.shape == (3, 2, 0)
        np.testing.assert_equal(t.numpy(), np.full((3, 2, 0), 12))
