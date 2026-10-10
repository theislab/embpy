"""Deferred module imports for optional heavy dependencies.

``embpy.embedder`` and ``embpy.models.base`` are part of the core surface --
``import embpy.embedder`` has to work on the lightweight install
(``pip install embpy``, no torch), because the resolvers, IO, annotation and
analysis layers all live behind it. Both modules nonetheless reference
``torch`` freely: 21 methods between them call ``torch.no_grad()``,
``isinstance(x, torch.Tensor)``, ``torch.nn.ModuleList`` and friends.

Every one of those references is inside a function body -- nothing in embpy
touches ``torch`` at class-definition, decorator, or module scope. So the
module-level ``import torch`` can be replaced by the proxy below without
changing a single call site: attribute access on the proxy performs the real
import, the first time anyone actually needs a tensor.

Deliberately *not* raising a custom error when the module is missing: the
natural ``ModuleNotFoundError("No module named 'torch'")`` is what
``embpy.embedder._missing_package_from_exception`` walks to turn a failed
backend import into a ``pip install`` line the user can run.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


class LazyModule(ModuleType):
    """A stand-in for ``import <name>`` that defers the import to first use.

    Behaves like the real module for attribute access, which is the only way
    embpy uses ``torch``. Resolved attributes are cached on the proxy, so the
    indirection costs one extra ``__getattr__`` per distinct attribute and
    nothing thereafter.

    Note this is a *proxy*, not the module itself: ``sys.modules`` is
    untouched, so ``import torch`` elsewhere still binds the real module and
    the two never diverge.

    Examples
    --------
    >>> torch = LazyModule("torch")  # no import happens here
    >>> torch.zeros(1).shape  # doctest: +SKIP
    torch.Size([1])
    """

    #: Protocol hooks that get probed *implicitly* -- by copy/pickle, by the
    #: import system, by IPython/pytest introspection. Answering them with
    #: AttributeError (rather than resolving the module) keeps a merely
    #: `repr()`-ed or `hasattr()`-probed proxy from importing torch, and keeps
    #: those probes returning False instead of raising ModuleNotFoundError
    #: when torch is genuinely absent. `__version__` and friends are NOT here:
    #: those are real attribute reads and must resolve.
    _NO_RESOLVE = frozenset(
        {
            "__path__",
            "__all__",
            "__copy__",
            "__deepcopy__",
            "__getstate__",
            "__setstate__",
            "__reduce__",
            "__reduce_ex__",
            "__getnewargs__",
            "__wrapped__",
            "__bases__",
            "__mro_entries__",
            "__iter__",
            "__next__",
            "__len__",
            "__await__",
            "_ipython_canary_method_should_not_exist_",
            "_ipython_display_",
            "_repr_html_",
            "_repr_mimebundle_",
            "_pytest_wrapped_",
        }
    )

    def __init__(self, name: str) -> None:
        super().__init__(name, f"Lazy proxy for the {name!r} module (embpy._lazy).")
        self._lazy_name = name

    def __getattr__(self, attr: str) -> Any:
        # `_lazy_name` lives in __dict__, so reaching it here means the proxy
        # was built without __init__ (unpickling, __class__ reassignment).
        # Guard explicitly: the alternative is infinite recursion.
        name = self.__dict__.get("_lazy_name")
        if name is None or attr in self._NO_RESOLVE:
            raise AttributeError(attr)
        value = getattr(import_module(name), attr)
        setattr(self, attr, value)
        return value

    def __repr__(self) -> str:
        return f"<LazyModule {self._lazy_name!r}>"


def lazy_module(name: str) -> Any:
    """Return a :class:`LazyModule` proxy for ``name``.

    Typed as ``Any`` so type checkers fall back to the ``if TYPE_CHECKING:
    import <name>`` declaration at the call site rather than checking against
    ``ModuleType``.
    """
    return LazyModule(name)


__all__ = ["LazyModule", "lazy_module"]
