"""Synthetic interaction-order tasks (PARITY-k, MAJORITY-k).

Kept out of ``tasks/`` on purpose: that package's ``__init__`` imports the
protein task classes, which pull in scipy and the TAPE data stack.  These tasks
need only numpy and torch, and are meant to stay runnable in a bare environment
such as a fresh Colab VM.
"""

from .order_tasks import (DEFAULT_CFG, TinyModel, build_model, centers_for,
                          chance_level, epochs_to, make_data, make_offsets,
                          n_params, run_one, set_seed)

__all__ = ["DEFAULT_CFG", "TinyModel", "build_model", "centers_for",
           "chance_level", "epochs_to", "make_data", "make_offsets",
           "n_params", "run_one", "set_seed"]
