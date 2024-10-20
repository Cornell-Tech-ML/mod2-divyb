"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    """Base class for all tensor operations in the autodifferentiation framework."""
    
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Perform the backward pass for the function."""
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Perform the forward pass for the function."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negation function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negation of the input tensor.

        Args:
            ctx (Context): The context for storing information.
            t1 (Tensor): The input tensor to negate.

        Returns:
            Tensor: The negated tensor.
            
        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the negation operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tensor: The gradient of the input tensor.
            
        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Inverse function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the inverse of the input tensor."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the inverse operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tensor: The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Addition function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the sum of two input tensors."""
        ctx.save_for_backward(t1, t2)
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the addition operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, Tensor]: The gradients of the input tensors.

        """
        return grad_output, grad_output


class All(Function):
    """Function to check if all elements are true."""
    
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all elements are true along the specified dimension."""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    """Multiplication function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the product of two input tensors."""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the multiplication operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, Tensor]: The gradients of the input tensors.

        """
        t1, t2 = ctx.saved_values
        return (
            grad_output.f.mul_zip(grad_output, t2),
            grad_output.f.mul_zip(t1, grad_output),
        )


class Sigmoid(Function):
    """Sigmoid activation function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the sigmoid of the input tensor."""
        sig = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the sigmoid operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tensor: The gradient of the input tensor.

        """
        (sig,) = ctx.saved_values
        neg_sig = sig.f.neg_map(sig)
        return grad_output.f.mul_zip(
            sig.f.mul_zip(sig, 1 + neg_sig),
            grad_output,
        )


class ReLU(Function):
    """ReLU activation function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Compute the ReLU of the input tensor."""
        ctx.save_for_backward(t)
        return t.f.relu_map(t)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the ReLU operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tensor: The gradient of the input tensor.

        """
        (t,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t, grad_output)


class Log(Function):
    """Logarithm function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the logarithm of the input tensor."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the logarithm operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tensor: The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """Exponential function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the exponential of the input tensor."""
        exp = t1.f.exp_map(t1)
        ctx.save_for_backward(exp)
        return exp

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the exponential operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tensor: The gradient of the input tensor.
            
        """
        (t,) = ctx.saved_values
        return grad_output.f.mul_zip(t, grad_output)


class Sum(Function):
    """Sum function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: int) -> Tensor:
        """Compute the sum of the input tensor along the specified dimension."""
        ctx.save_for_backward(t1.shape, dim)
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the sum operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, float]: The gradient of the input tensor and a placeholder.

        """
        _, _ = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    """Function to check if all elements are true along a specified dimension."""
    
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    """Less than comparison function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise less than comparison between two tensors."""
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the less than operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, Tensor]: The gradients of the input tensors.

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    """Equality comparison function for tensors."""
    
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise equality comparison between two tensors."""
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the equality operation.

        Args:
            ctx (Context): The context for storing information.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, Tensor]: The gradients of the input tensors.

        """
        t1_shape, t2_shape = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class IsClose(Function):
    """Function to check if two tensors are close to each other within a tolerance."""

    @staticmethod
    def forward(
        ctx: Context, t1: Tensor, t2: Tensor, atol: float = 1e-8, rtol: float = 1e-5
    ) -> Tensor:
        """Check if two tensors are close within a specified absolute and relative tolerance.

        Args:
            ctx (Context): The context for storing information.
            t1 (Tensor): The first tensor to compare.
            t2 (Tensor): The second tensor to compare.
            atol (float): The absolute tolerance.
            rtol (float): The relative tolerance.

        Returns:
            Tensor: A tensor indicating whether each element of t1 and t2 are close.
            
        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.is_close_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward function"""        
        t1_shape, t2_shape = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class Permute(Function):
    """Permute the dimensions of a tensor based on a specified order."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of a tensor, given a new order."""
        # permute(self, *order:int) is a method defined for TensorData in tensor_data.py
        # Tensor._tensor is an instance of TensorData
        # TensorData._storage is of type Storage (tensor_data.py), a np array of float64
        # So we must convert to a list of integers to pass as `order`
        int_order = [int(x) for x in order._tensor._storage]
        ctx.save_for_backward(int_order)

        # t1._new creates a new tensor with the same backend as `t1`
        return t1._new(t1._tensor.permute(*int_order))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Permute the gradients back to the original order."""
        (new_order,) = ctx.saved_values

        # Create a mapping from value to index for efficient lookup
        # Use the mapping to construct the original order
        original_order_map = {v: i for i, v in enumerate(new_order)}
        original_order = [original_order_map[i] for i in range(len(new_order))]

        return grad_output._new(grad_output._tensor.permute(*original_order)), 0.0


class View(Function):
    """View function for reshaping tensors."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    """Identity function for tensors, returning the input tensor as is."""
    
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    """Matrix multiplication function for tensors."""
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )
        


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference approximation of the gradient.

    Args:
        f (Any): The function for which the gradient is being computed.
        *vals (Tensor): The input tensors to the function.
        arg (int): The index of the argument to compute the gradient for.
        epsilon (float): The small perturbation used for the central difference.
        ind (UserIndex): The index at which to compute the gradient.

    Returns:
        float: The approximate gradient at the specified index.
        
    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )


