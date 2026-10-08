"""LDC-YOLO modules and registration helpers."""
from .blocks import (
    DLStem, LGDI, C2fLGDI, RPMKRepViT, C2fRPMK, GhostBottleneckV2,
    VisDroneDCNDetect, fuse_all_reparam_modules,
    ConvStem, C2f_E, C2f_G,
)
from .register import register_ldc_yolo
register_mkl_yolo = register_ldc_yolo
