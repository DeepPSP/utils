# -*- coding: utf-8 -*-
"""
docstring, to write
"""

from functools import wraps


__all__ = [
    "indicator_enter_leave_func",
    "trivial_jit",
]


def indicator_enter_leave_func(verbose:int=0):
    """

    Parameters:
    -----------
    verbose: int,
        printing verbosity
    """
    nl = "\n"
    def dec_outer(fn:callable):
        @wraps(fn)
        def dec_inner(*args, **kwargs):
            if verbose >= 1:
                print(f"{nl}{'*'*6}  entering function {fn.__name__}  {'*'*6}")
                start = time.time()
            response = fn(*args, **kwargs)
            if verbose >= 1:
                print(f"{nl}{'*'*6}  execution of function {fn.__name__} used {time.time()-start} second(s)  {'*'*6}")
                print(f"{nl}{'*'*6}  leaving function {fn.__name__}  {'*'*6}{nl}")
            return response
        return dec_inner
    return dec_outer


def trivial_jit(signature_or_function=None, locals={}, target='cpu', cache=False, pipeline_class=None, **options):
    """

    Parameters:
    -----------
    ref. `numba.jit`
    """
    def dec(fn:callable):
        return fn
    return dec
