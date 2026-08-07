"""Path shim for the sibling data package — the ONE place that touches ``sys.path``.

``synthdata`` lives in ``../data`` and is not installed by default. Importing this
module prepends ``<repo>/data`` to ``sys.path`` so ``import synthdata`` works with
zero install friction. If a real ``synthdata`` installation is already importable it
wins and the shim does nothing.

Every module here that needs ``synthdata`` imports this module *first*::

    import _paths  # noqa: F401  (sys.path shim for synthdata)
    from synthdata.tokenizer import BitTokenizer
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: The ``model/`` directory. Relative paths in configs are resolved against it, so a
#: config means the same thing no matter which directory the CLI is invoked from.
MODEL_ROOT = Path(__file__).resolve().parent


def _install_synthdata_path_shim() -> Path | None:
    """Make ``import synthdata`` work from the sibling ``data/`` directory."""
    if importlib.util.find_spec("synthdata") is not None:
        return None
    data_dir = MODEL_ROOT.parent / "data"
    if (data_dir / "synthdata" / "__init__.py").exists():
        sys.path.insert(0, str(data_dir))
        return data_dir
    return None


SYNTHDATA_PATH = _install_synthdata_path_shim()
