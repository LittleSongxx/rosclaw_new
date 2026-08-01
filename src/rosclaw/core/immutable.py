"""Read-only payload wrappers for EventBus hardening.

Subscribers receive payloads wrapped in these classes.  They subclass
``dict``/``list`` so ``json.dumps`` and every read-path consumer keeps
working unchanged, but every Python-level mutation entry point raises
``TypeError`` — one subscriber can never alter what another subscriber
(or the publisher's history) observes.
"""

from __future__ import annotations

from typing import Any

_MESSAGE = (
    "Event payload is read-only for subscribers (EventBus hardening): "
    "derive your own copy (dict(x) / list(x) / copy.deepcopy) instead of mutating it"
)


class FrozenDict(dict):
    """dict that refuses mutation."""

    __slots__ = ()

    def _blocked(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(_MESSAGE)

    __setitem__ = _blocked
    __delitem__ = _blocked
    pop = _blocked
    popitem = _blocked
    clear = _blocked
    update = _blocked
    setdefault = _blocked
    __ior__ = _blocked

    def copy(self) -> dict:
        import copy as _copy

        return _copy.deepcopy(self)

    def __deepcopy__(self, memo: dict) -> dict:
        """Deep-copy into a PLAIN dict (copy module rebuilds via append/setitem,
        which are blocked on the frozen wrapper)."""
        import copy as _copy

        return {k: _copy.deepcopy(v, memo) for k, v in self.items()}


class FrozenList(list):
    """list that refuses mutation."""

    __slots__ = ()

    def _blocked(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(_MESSAGE)

    __setitem__ = _blocked
    __delitem__ = _blocked
    append = _blocked
    extend = _blocked
    insert = _blocked
    remove = _blocked
    pop = _blocked
    clear = _blocked
    sort = _blocked
    reverse = _blocked
    __iadd__ = _blocked
    __imul__ = _blocked

    def copy(self) -> list:
        import copy as _copy

        return _copy.deepcopy(self)

    def __deepcopy__(self, memo: dict) -> list:
        import copy as _copy

        return [_copy.deepcopy(v, memo) for v in self]


def freeze(value: Any) -> Any:
    """Return a read-only recursive wrapper of value (idempotent)."""
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, dict):
        return FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value: Any) -> Any:
    """Return a plain mutable deep copy of a (possibly frozen) value."""
    import copy as _copy

    return _copy.deepcopy(value)


# PyYAML's SafeRepresenter uses exact-type lookup; register the frozen
# subclasses so practice/ledger YAML writers keep working transparently.
try:
    import yaml

    yaml.SafeDumper.add_multi_representer(
        FrozenDict, yaml.SafeDumper.represent_dict
    )
    yaml.SafeDumper.add_multi_representer(
        FrozenList, yaml.SafeDumper.represent_list
    )
except ImportError:  # pragma: no cover - pyyaml is a core dependency
    pass
