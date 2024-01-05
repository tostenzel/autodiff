import numpy as np
import unittest, copy
from edugrad import Tensor
from edugrad.dtypes import dtypes


# Tensor(x) casts all types up to float32
class TestEdugrad(unittest.TestCase):
    def test_ndim(self):
        assert Tensor(1).ndim == 0
        assert Tensor.randn(1).ndim == 1
        assert Tensor.randn(2, 2, 2).ndim == 3
        assert Tensor.randn(1, 1, 1, 1, 1, 1).ndim == 6

    def test_numel(self):
        assert Tensor.randn(10, 10).numel() == 100
        assert Tensor.randn(1, 2, 5).numel() == 10
        assert Tensor.randn(1, 1, 1, 1, 1, 1).numel() == 1
        assert Tensor([]).numel() == 0
        assert Tensor.randn(1, 0, 2, 5).numel() == 0

    def test_element_size(self):
        for _, dtype in dtypes.fields().items():
            assert (
                dtype.itemsize == Tensor.randn(3, dtype=dtype).element_size()
            ), f"Tensor.element_size() not matching Tensor.dtype.itemsize for {dtype}"

    def test_zerosized_tensors(self):
        np.testing.assert_equal(Tensor([]).numpy(), np.array([]))
        np.testing.assert_equal(Tensor(None).numpy(), np.array([]))

    def test_tensor_ndarray_dtype(self):
        arr = np.array([1])  # where dtype is implicitly int64
        with self.assertRaises(KeyError):
            # DTYPES_DICT[np.dtype(x).name] key not available because dtype.int64 not defined
            assert Tensor(arr).dtype == dtypes.int64
        assert (
            Tensor(arr, dtype=dtypes.float32).dtype == dtypes.float32
        )  # check if ndarray correctly casts to Tensor dtype
        with self.assertRaises(AttributeError):
            # dtype.float64 not defined
            assert Tensor(arr, dtype=dtypes.float64).dtype == dtypes.float64  # check that it works for something else

    def test_tensor_list_dtype(self):
        for arr in ([1], [[[1]]], [[1, 1], [1, 1]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]):
            with self.assertRaises(AssertionError):
                # we always cast up to float32, even only int int32.
                assert Tensor(arr).dtype == dtypes.only_int
            assert Tensor(arr, dtype=dtypes.float32).dtype == dtypes.float32

        for arr in (
            [True],
            [[[False]]],
            [[True, False], [True, False]],
            [[[False, True], [False, False]], [[True, True], [False, True]]],
        ):
            with self.assertRaises(AssertionError):
                # we always cast up to float32, even bool.
                assert Tensor(arr).dtype == dtypes.bool
            assert Tensor(arr, dtype=dtypes.float32).dtype == dtypes.float32
            with self.assertRaises(AttributeError):
                # dtype.float64 not defined
                assert Tensor(arr, dtype=dtypes.float64).dtype == dtypes.float64

        # empty tensor defaults
        for arr in ([], [[[]]], [[], []]):
            t = Tensor(arr)
            assert t.dtype == dtypes.only_float
            np.testing.assert_allclose(t.numpy(), np.array(arr))

        # mixture of bool and int
        for arr in ([True, 3], [[True], [3]], [[[True]], [[3]]], [[True, 3], [3, True]]):
            t = Tensor(arr)
            with self.assertRaises(AssertionError):
                # we always cast up to float32,
                assert t.dtype == dtypes.only_int
            np.testing.assert_allclose(t.numpy(), np.array(arr))

        # mixture of bool, int and float
        for arr in (
            [[True, True], [3.0, True]],
            [[0, 1], [3.0, 4]],
            [[[0], [1]], [[3.0], [4]]],
            [[[True], [1]], [[3.0], [4]]],
        ):
            t = Tensor(arr)
            assert t.dtype == dtypes.only_float
            np.testing.assert_allclose(t.numpy(), np.array(arr))

    def test_tensor_list_shapes(self):
        self.assertEqual(Tensor([[[]]]).shape, (1, 1, 0))
        self.assertEqual(Tensor([[], []]).shape, (2, 0))
        self.assertEqual(Tensor([[[[]], [[]]], [[[]], [[]]], [[[]], [[]]]]).shape, (3, 2, 1, 0))

    def test_tensor_list_errors(self):
        # inhomogeneous shape
        with self.assertRaises(ValueError):
            Tensor([[], [[]]])
        with self.assertRaises(ValueError):
            Tensor([[1], []])
        with self.assertRaises(ValueError):
            Tensor([[1], [1], 1])
        with self.assertRaises(ValueError):
            Tensor([[[1, 1, 1], [1, 1]]])
        with self.assertRaises(ValueError):
            Tensor([[1, 1, 1], [[1, 1, 1]]])

    def test_tensor_copy(self):
        x = copy.deepcopy(Tensor.ones((3, 3, 3)))
        np.testing.assert_allclose(x.numpy(), np.ones((3, 3, 3)))

    def test_item_to_tensor_to_item(self):
        for a in [0, 1, 2, 3, -1, -100, 100, -101.1, 2.345, 100.1, True, False]:
            tensor_item = Tensor(a).item()
            buffered_tensor_item = Tensor([a]).item()
            reshaped_tensor_item = Tensor([a]).reshape((1, 1, 1, 1, 1)).item()
            np.testing.assert_allclose(tensor_item, a)
            np.testing.assert_allclose(buffered_tensor_item, a)
            np.testing.assert_allclose(reshaped_tensor_item, a)
            self.assertEqual(type(tensor_item), type(a))

            # For non-floats, assert that type check raises AssertionError if Tensor created from list
            if isinstance(a, float):
                # For floats, type should be retained
                self.assertEqual(type(tensor_item), float)
                self.assertEqual(type(buffered_tensor_item), float)
                self.assertEqual(type(reshaped_tensor_item), float)
            else:
                with self.assertRaises(AssertionError):
                    self.assertEqual(type(buffered_tensor_item), type(a))
                with self.assertRaises(AssertionError):
                    self.assertEqual(type(reshaped_tensor_item), type(a))


class TestZeroShapeTensor(unittest.TestCase):
    def test_elementwise(self):
        a = Tensor.rand(3, 2, 0)
        a_exp = a.exp()
        assert a_exp.shape == (3, 2, 0)
        np.testing.assert_equal(a_exp.numpy(), np.exp(a.numpy()))

        b = Tensor.rand(3, 2, 0)
        assert b.shape == (3, 2, 0)
        ab = a * b
        assert ab.shape == (3, 2, 0)
        np.testing.assert_equal(ab.numpy(), a.numpy() * b.numpy())

        mask = Tensor.rand(3, 2, 0) > 0.5
        assert mask.shape == (3, 2, 0)
        c = mask.where(a, b)
        assert c.shape == (3, 2, 0)
        np.testing.assert_equal(c.numpy(), np.where(mask.numpy(), a.numpy(), b.numpy()))


if __name__ == "__main__":
    unittest.main()
