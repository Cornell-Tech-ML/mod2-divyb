from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol

# Task 1.1
# Central Difference calculation


def central_difference(f: Any, vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    """Computes an approximation to the derivative of `f` with respect to one arg.

    See `doc: derivate` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f: arbitrary function from n-scalar args to one value
        vals: n-float values $(x_0 \ldots x_{n-1})$
        arg: the number $i$ of the arg to compute the derivative
        epsilon: a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0 \ldots x_{n-1})$

    """
    val1 = [v for v in vals]
    val2 = [v for v in vals]
    val1[arg] = vals[arg] + epsilon
    val2[arg] = vals[arg] - epsilon
    return (f(*val1) - f(*val2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    @property
    def unique_id(self) -> int: ...

    def is_leaf(self) -> bool: ...

    def is_constant(self) -> bool: ...

    @property
    def parents(self) -> Iterable[Variable]: ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for v in var.parents:
                if not v.is_constant():
                    visit(v)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
       compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
