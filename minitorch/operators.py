"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers or tensors element-wise.

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The element-wise product of `x` and `y`.

    """
    return x * y


def id(x: float) -> float:
    """Returns input unchanged.

    Args:
    ----
        x: The first number or tensor.

    Returns:
    -------
        The input unchanged

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers or tensors element-wise.

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The element-wise addition of `x` and `y`.

    """
    return x + y


def neg(x: float) -> float:
    """Returns negation of the input.

    Args:
    ----
        x: The first number or tensor.

    Returns:
    -------
        The input negated

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The true if one number is less than another of `x` and `y`.

    """
    return x < y


def eq(x: float, y: float) -> float:
    """Checks if one number is equal to another

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The true if one number equal to another of `x` and `y`.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Checks max

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The larger of `x` and `y`.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are equal

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The larger of `x` and `y`.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Returns sigmoid of the input.

    Args:
    ----
        x: The first number or tensor.

    Returns:
    -------
        The sigmoid of x

    """
    # $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    if x >= 0:
        out = 1 / (1 + (math.exp(x * -1)))
    else:
        out = math.exp(x) / (1 + (math.exp(x)))
    return out


def relu(x: float) -> float:
    """Returns relu of the input.

    Args:
    ----
        x: The first number or tensor.

    Returns:
    -------
        The relu of x

    """
    return max(0, x)


def log(x: float) -> float:
    """Returns log of the input.

    Args:
    ----
        x: The first number or tensor.

    Returns:
    -------
        The log of x

    """
    return math.log(x)


def exp(x: float) -> float:
    """Returns exp of the input.

    Args:
    ----
        x: The first number or tensor.

    Returns:
    -------
        The exp of x

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Returns inv of the input.

    Args:
    ----
        x: The first number or tensor.

    Returns:
    -------
        The inv of x

    """
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Returns inv of the input.

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The inv of x

    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Returns inv_back of the input.

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The inv_back of the input

    """
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Returns relu_back of the input.

    Args:
    ----
        x: The first number or tensor.
        y: The second number or tensor.

    Returns:
    -------
        The relu_back of the input

    """
    return y if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce


def map(func: Callable[[float], float], iterable: list[float]) -> list[float]:
    """Apply a function to each element of a list and return a new list with the results.

    This function takes a function and an iterable of floats, applies the function
    to each element of the iterable, and returns a new list containing the results.

    Parameters
    ----------
    func : Callable[[float], float]
        A function that takes a single float argument and returns a float.
    iterable : List[float]
        A list of floats to which the function will be applied.

    Returns
    -------
    List[float]
        A list of floats resulting from applying the function to each element
        of the input list.

    """
    lst = []
    for i in iterable:
        lst.append(func(i))
    return lst


def zipWith(
    func: Callable[[float, float], float],
    iterable1: Iterable[float],
    iterable2: Iterable[float],
) -> list[float]:
    """Combine elements from two lists using a function.

    This function takes a binary function and two lists of floats, applies the function
    to corresponding elements from both lists, and returns a new list with the results.

    Parameters
    ----------
    func : Callable[[float, float], float]
        A function that takes two float arguments and returns a float.
    iterable1 : List[float]
        The first list of floats.
    iterable2 : List[float]
        The second list of floats.

    Returns
    -------
    List[float]
        A list of floats resulting from applying the function to corresponding
        elements from both input lists.

    """
    iter1 = iter(iterable1)
    iter2 = iter(iterable2)

    lst = []
    while True:
        try:
            lst.append(func(next(iter1), next(iter2)))
        except StopIteration:
            break
    return lst


def reduce(func: Callable[[float, float], float], iterable: Iterable[float]) -> float:
    """Returns mapped of a function to list

    Args:
    ----
        func: The function.
        iterable: The second number or tensor.

    Returns:
    -------
        Higher-order function that reduces an iterable to a single value using a given function

    """
    iterator = iter(iterable)

    # Initialize the result with the first element
    try:
        result = next(iterator)
    except StopIteration:
        return 0

    # Apply the reduction operation for the rest of the elements
    for item in iterator:
        result = func(result, item)

    return result


# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def negList(lst: list[float]) -> list[float]:
    """Negates each element in the list.

    Args:
    ----
        lst (list[float]): A list of floats to be negated.

    Returns:
    -------
        list[float]: A list with each element negated.

    """
    return map(neg, lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> list[float]:
    """Add corresponding elements of two iterables using the `zipWith` function.

    This function takes two iterables of floats and returns a list of floats where
    each element is the sum of corresponding elements from the input iterables.

    Parameters
    ----------
    lst1 : Iterable[float]
        The first iterable of floats.
    lst2 : Iterable[float]
        The second iterable of floats.

    Returns
    -------
    List[float]
        A list of floats resulting from adding corresponding elements from `lst1` and `lst2`.

    """
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Negates each element in the list.

    Args:
    ----
        lst (list[float]): A list of floats to be summed.

    Returns:
    -------
        sum of elements of lst

    """
    return reduce(add, lst)


def prod(lst: Iterable[float]) -> float:
    """Negates each element in the list.

    Args:
    ----
        lst (list[float]): A list of floats to be summed.

    Returns:
    -------
        sum of elements of lst

    """
    return reduce(mul, lst)


# TODO: Implement for Task 0.3.
