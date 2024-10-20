from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple
    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """Scalar wrapper for a mathematical function that processes and produces Scalar variables.

    This is a static class and is never instantiated. We use 'class'
    here to group together the 'forward' and 'backward' code
    """

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the output with respect to the inputs."""
        return wrap_tuple(cls.backward(ctx, d_output))  # type: ignore

    @classmethod
    def forward(cls, ctx: Context, *inputs: float) -> float:
        """Compute the output of the function given the inputs."""
        return cls.forward(ctx, *inputs)  # type: ignore

    @classmethod
    def apply(cls, vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values and return a Scalar."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        ctx = Context(vals)
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))
        # Create a new variable for the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Example
class Add(ScalarFunction):
    """Add function."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the gradient of the output with respect to the inputs for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the output with respect to the inputs for addition."""
        return d_output, d_output


# Log function
class Log(ScalarFunction):
    """Logarithm function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the logarithm of a given value."""
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the logarithm with respect to the input."""
        (a,) = ctx.saved_values
        return (d_output / a,)


# Multiply function
class Mul(ScalarFunction):
    """Multiplication function."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the product of two values."""
        assert a
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradients of the product with respect to the inputs."""
        a, b = ctx.saved_values
        return b * d_output, a * d_output


# Inverse function
class Inv(ScalarFunction):
    """Inverse function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of a given value."""
        assert a
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the inverse with respect to the input."""
        (a,) = ctx.saved_values
        return -d_output / (a**2)


# Negation function
class Neg(ScalarFunction):
    """Negation function."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negate the input value."""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the negation with respect to the input."""
        return -d_output


# Sigmoid function
class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid of a given value."""
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the sigmoid with respect to the input."""
        out = ctx.saved_values[0]
        return (d_output * out * (1 - out),)


# ReLU function
class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the ReLU of a given value."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the output with respect to the inputs."""
        (a,) = ctx.saved_values
        return d_output * (a > 0)


# Exp function
class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of a given value."""
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the output with respect to the inputs."""
        out = ctx.saved_values[0]
        return (d_output * out,)


# LessThan function
class LT(ScalarFunction):
    """Less-than function $f(a) = 1.0 if x is less than y else 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the less-than comparison between two values."""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the output with respect to the inputs."""
        return 0.0, 0.0


# Equal function
class EQ(ScalarFunction):
    """Equal function $f(a) = 1.0 if x is equal to y else 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the equality comparison between two values."""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the output with respect to the inputs."""
        return 0.0, 0.0
