"""Import pure serving modules without requiring the compiled Mirage extension."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name, path in [("mirage", ROOT / "python/mirage"),
                   ("mirage.engine", ROOT / "python/mirage/engine")]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules.setdefault(name, module)
