"""Cargar métricas/plots existentes sin ejecutar los __init__ del pipeline completo.

evaluation/__init__.py importa report -> lstm -> feature_preparation.data,
un módulo ausente en este checkout. Un namespace privado permite reutilizar
los archivos originales y sus imports relativos sin editarlos ni copiar lógica.
"""

from importlib import import_module
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType


def cargar_original(nombre):
    root = Path(__file__).resolve().parents[1] / "models"
    prefix = f"{__package__}._original"
    for name, path in [(prefix, root), (f"{prefix}.evaluation", root / "evaluation")]:
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            package.__spec__.submodule_search_locations = package.__path__
            sys.modules[name] = package
    return import_module(f"{prefix}.evaluation.{nombre}")
