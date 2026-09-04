"""Synthetic interaction-order tasks.

Two families, kept out of ``tasks/`` on purpose: that package's ``__init__``
imports the protein task classes, which pull in scipy and the TAPE data stack.
These need only numpy and torch, and are meant to stay runnable in a bare
environment such as a fresh Colab VM.

``order_tasks``   PARITY-k / MAJORITY-k -- interaction order at fixed access.
``match_tasks``   MATCH-q on Z_M -- the task family of Sanford et al. (2024),
                  generalised to any order.

Names common to both modules (``run_one``, ``make_data``, ``build_model``,
``TinyModel``) are exported unprefixed from ``order_tasks`` and with a
``match_`` prefix from ``match_tasks``.
"""

from .order_tasks import (DEFAULT_CFG, TinyModel, build_model, centers_for,
                          chance_level, epochs_to, make_data, make_offsets,
                          n_params, run_one, set_seed)

from .match_tasks import (TinyModel as MatchTinyModel, auto_batch, base_rate,
                          build_model as match_build_model, calibrate_M,
                          full_window, fourier_table,
                          make_data as match_make_data, match_labels,
                          run_one as match_run_one, tuple_count)

__all__ = ["DEFAULT_CFG", "TinyModel", "build_model", "centers_for",
           "chance_level", "epochs_to", "make_data", "make_offsets",
           "n_params", "run_one", "set_seed",
           # match-q
           "MatchTinyModel", "auto_batch", "base_rate", "match_build_model",
           "calibrate_M", "full_window", "fourier_table", "match_make_data",
           "match_labels", "match_run_one", "tuple_count"]
