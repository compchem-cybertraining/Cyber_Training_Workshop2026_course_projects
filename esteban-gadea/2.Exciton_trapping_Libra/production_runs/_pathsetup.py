"""Shared path setup for the production run/analysis scripts in this folder.

Import this before any `model`/`recipes` import, e.g.:

    import _pathsetup  # noqa: F401
    from cis_compute_adi import cis_compute_adi, get_default_params, ring_positions
    from recipes import ehrenfest_onthefly

Adds `2.Exciton_trapping_Libra/model/` (for the direct `cis_compute_adi`-style
imports the model modules use internally) and `2.Exciton_trapping_Libra/` itself
(for the namespace-package import `from recipes import ehrenfest_onthefly`) to
sys.path, relative to this file's location -- works no matter what directory the
script is actually invoked from.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.dirname(_here)  # .../2.Exciton_trapping_Libra
for _p in (_base, os.path.join(_base, "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
