"""Runtime registration for MKL-YOLO modules in Ultralytics."""

from __future__ import annotations

from typing import Iterable

from .blocks import C2f_E, C2f_G, ConvStem, GhostBottleneckV2, VisDroneDCNDetect


CUSTOM_MODULES = (ConvStem, GhostBottleneckV2, C2f_E, C2f_G, VisDroneDCNDetect)


def _patch_parse_model() -> None:
    """Patch Ultralytics parser sets so MKL modules are parsed like native modules."""
    import ultralytics.nn.tasks as tasks

    for module in CUSTOM_MODULES:
        setattr(tasks, module.__name__, module)

    original_parse_model = tasks.parse_model
    if getattr(original_parse_model, "_mkl_yolo_patched", False):
        return

    def parse_model_with_mkl(d, ch, verbose=True):
        source = original_parse_model.__globals__
        for module in CUSTOM_MODULES:
            source[module.__name__] = module

        # Ultralytics keeps base/repeat module sets local to parse_model. The
        # maintained fork in this repository already includes these modules. For
        # pip-installed Ultralytics, fall back to that parser when the version has
        # native support; otherwise provide a clear error message.
        try:
            return original_parse_model(d, ch, verbose)
        except KeyError as err:
            missing = err.args[0]
            if missing in {m.__name__ for m in CUSTOM_MODULES}:
                raise RuntimeError(
                    "This Ultralytics version does not expose custom module parsing hooks. "
                    "Use the patched Ultralytics code in this repository or install a version "
                    "whose parse_model includes MKL-YOLO modules in base_modules/repeat_modules."
                ) from err
            raise

    parse_model_with_mkl._mkl_yolo_patched = True
    tasks.parse_model = parse_model_with_mkl


def _patch_module_exports() -> None:
    """Expose MKL classes from ultralytics.nn.modules for YAML lookup."""
    import ultralytics.nn.modules as modules

    for module in CUSTOM_MODULES:
        setattr(modules, module.__name__, module)


def register_mkl_yolo(extra_modules: Iterable[type] = ()) -> None:
    """Register MKL-YOLO modules before constructing an Ultralytics YOLO model."""
    _patch_module_exports()
    _patch_parse_model()
    if extra_modules:
        import ultralytics.nn.tasks as tasks

        for module in extra_modules:
            setattr(tasks, module.__name__, module)


register_mkl_yolo()
