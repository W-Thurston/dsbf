# dsbf/checks/__init__.py
"""The ``core`` pack: the checks DSBF ships.

A check is declared with ``@check`` at the top level of its own module here, so a
worker process can import it by name, and ``core_pack`` lists it. Nothing registers
on import: ``dsbf.profile()`` and ``dsbf plan`` call ``Registry.register_pack``.

Conventions every core check follows:

- A check reads the columns ``applies_to`` selects, through ``ctx.columns``, the
  profile and ``ctx.values``, and never picks columns by dtype itself.
- A check never calls ``warnings.catch_warnings()``. The engine records a warning a
  check raises as a ``python_warning`` note with a count; the block would also hide
  the warnings of the other checks running beside it.
- A check gives scipy numeric arrays, never object dtype: scipy checks an object
  array for NaN one element at a time.
- A check states ``n_used`` and the method on every finding's evidence, declares the
  bounds of each parameter, and leaves recommendations to the tips attached after it
  returns.
"""

from dsbf.checks.zeros import zeros
from dsbf.engine import Pack

CORE_PACK = "core"

__all__ = ["CORE_PACK", "core_pack"]


def core_pack() -> Pack:
    """The core checks, as a pack to register explicitly."""
    return Pack(CORE_PACK, (zeros,))
