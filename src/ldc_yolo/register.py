"""Register LDC modules with the pinned Ultralytics 8.3.0 parser."""
from __future__ import annotations
import ast
import inspect
import textwrap
from .blocks import DLStem, C2fLGDI, C2fRPMK, GhostBottleneckV2, VisDroneDCNDetect

CUSTOM_MODULES = (DLStem, GhostBottleneckV2, C2fLGDI, C2fRPMK, VisDroneDCNDetect)
ALIASES = {"ConvStem": DLStem, "C2f_E": C2fLGDI, "C2f_G": C2fRPMK}

def register_ldc_yolo():
    import ultralytics
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    if ultralytics.__version__ != "8.3.0":
        raise RuntimeError("LDC-YOLO first release requires ultralytics==8.3.0.")
    exports = {m.__name__: m for m in CUSTOM_MODULES} | ALIASES
    for name, module in exports.items():
        setattr(tasks, name, module)
        setattr(modules, name, module)
    if getattr(tasks.parse_model, "_ldc_yolo_patched", False):
        return
    # Extend the upstream parser's channel/repeat/head membership checks in memory.
    # No installed library files are modified.
    tree = ast.parse(textwrap.dedent(inspect.getsource(tasks.parse_model)))
    changed = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not any(isinstance(op, ast.In) for op in node.ops):
            continue
        for group in node.comparators:
            if not isinstance(group, (ast.Set, ast.Tuple, ast.List)):
                continue
            names = {x.id for x in group.elts if isinstance(x, ast.Name)}
            extra = []
            if {"Conv", "C2f"} <= names:
                extra = ["DLStem", "GhostBottleneckV2", "C2fLGDI", "C2fRPMK", *ALIASES]
                changed.add("channels")
            elif {"BottleneckCSP", "C2f"} <= names and "Conv" not in names:
                extra = ["C2fLGDI", "C2fRPMK", "C2f_E", "C2f_G"]
                changed.add("repeats")
            elif {"Detect", "Segment"} <= names:
                extra = ["VisDroneDCNDetect"]
                changed.add("heads")
            group.elts.extend(ast.Name(id=name, ctx=ast.Load()) for name in extra)
    if changed != {"channels", "repeats", "heads"}:
        raise RuntimeError("Unsupported Ultralytics parser layout.")
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), "<ldc-yolo-parser>", "exec"), tasks.__dict__, namespace)
    tasks.parse_model = namespace["parse_model"]
    tasks.parse_model._ldc_yolo_patched = True

register_mkl_yolo = register_ldc_yolo
register_ldc_yolo()
