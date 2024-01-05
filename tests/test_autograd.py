import numpy as np
import torch
import unittest
from edugrad import Tensor

from tests.gradcheck import numerical_jacobian, jacobian, gradcheck

x_init = np.random.randn(1, 3).astype(np.float32)
U_init = np.random.randn(3, 3).astype(np.float32)
V_init = np.random.randn(3, 3).astype(np.float32)
W_init = np.random.randn(3, 3).astype(np.float32)
m_init = np.random.randn(1, 3).astype(np.float32)


class TestEdugrad(unittest.TestCase):
    def test_backward_pass(self):
        def test_edugrad():
            x = Tensor(x_init, requires_grad=True)
            W = Tensor(W_init, requires_grad=True)
            m = Tensor(m_init)
            out = x.dot(W).relu()
            out = out.log_softmax()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.numpy(), x.grad.numpy(), W.grad.numpy()

        def test_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            m = torch.tensor(m_init)
            out = x.matmul(W).relu()
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, W.grad

        for x, y in zip(test_edugrad(), test_pytorch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_backward_pass_diamond_model(self):
        def test_edugrad():
            u = Tensor(U_init, requires_grad=True)
            v = Tensor(V_init, requires_grad=True)
            w = Tensor(W_init, requires_grad=True)
            x = u.mul(v).relu()
            y = u.mul(w).relu()
            out = x.add(y).mul(y).relu()
            out = out.log_softmax()
            out = out.sum()
            out.backward()
            return out.numpy(), u.grad.numpy(), v.grad.numpy(), w.grad.numpy()

        def test_pytorch():
            u = torch.tensor(U_init, requires_grad=True)
            v = torch.tensor(V_init, requires_grad=True)
            w = torch.tensor(W_init, requires_grad=True)
            x = u.mul(v).relu()
            y = u.mul(w).relu()
            out = x.add(y).mul(y).relu()
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), u.grad, v.grad, w.grad

        for x, y in zip(test_edugrad(), test_pytorch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_nograd(self):
        x = Tensor(x_init, requires_grad=False)
        m = Tensor(m_init, requires_grad=False)
        W = Tensor(W_init, requires_grad=True)
        tmp = x.mul(m)
        mm = tmp.matmul(W)
        out = mm.relu()
        out = out.sum()
        out.backward()
        assert x.grad is None
        assert m.grad is None
        assert tmp.grad is None
        assert mm.grad is not None
        assert W.grad is not None

    def test_jacobian(self):
        W = np.random.RandomState(42069).random((10, 5)).astype(np.float32)
        x = np.random.RandomState(69420).random((1, 10)).astype(np.float32)

        torch_x = torch.tensor(x, requires_grad=True)
        torch_W = torch.tensor(W, requires_grad=True)

        def torch_func(x):
            return torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)

        PJ = torch.autograd.functional.jacobian(torch_func, torch_x).squeeze().numpy()

        edugrad_x = Tensor(x, requires_grad=True)
        edugrad_W = Tensor(W, requires_grad=True)

        def func(x):
            return x.dot(edugrad_W).relu().log_softmax()

        J = jacobian(func, edugrad_x)
        NJ = numerical_jacobian(func, edugrad_x)

        np.testing.assert_allclose(PJ, J, atol=1e-5)
        np.testing.assert_allclose(PJ, NJ, atol=1e-3)

    def test_gradcheck(self):
        W = np.random.RandomState(1337).random((10, 5)).astype(np.float32)
        x = np.random.RandomState(7331).random((1, 10)).astype(np.float32)

        edugrad_x = Tensor(x, requires_grad=True)
        edugrad_W = Tensor(W, requires_grad=True)

        def func(x):
            return x.dot(edugrad_W).relu().log_softmax()

        self.assertTrue(gradcheck(func, edugrad_x, eps=1e-3))

        # coarse approx. since a "big" eps and the non-linearities of the model
        self.assertFalse(gradcheck(func, edugrad_x, eps=1e-5))

    def test_deepwalk_ctx_check(self):
        layer = Tensor.uniform(1, 1, requires_grad=True)
        x = Tensor.randn(1, 1, 1)
        x.dot(layer).mean().backward()
        x = Tensor.randn(1, 1, 1)
        x.dot(layer).mean().backward()
