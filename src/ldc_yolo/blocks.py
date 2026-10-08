"""Custom modules used by LDC-YOLO."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


def autopad(k: int, p: Optional[int] = None, d: int = 1) -> int:
    """Return padding that preserves spatial shape for stride 1."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class DLStem(nn.Module):
    """DL-Stem: 3x3 projection followed by stride-2 3x3 downsampling."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, mid_ratio: float = 1.0) -> None:
        super().__init__()
        c_mid = max(int(c2 * mid_ratio), 1)
        if isinstance(k, list) and len(k) == 2:
            k1, k2 = k
        else:
            k1, k2 = k, k
        self.conv1 = Conv(c1, c_mid, k1, s=1)
        self.conv2 = Conv(c_mid, c2, k2, s=s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class LGDI(nn.Module):
    """LGDI: fixed eight-neighbour Laplacian, BN, tanh(abs), DW/PW refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: Union[bool, nn.Module] = True,
        use_dwconv: bool = True,
    ) -> None:
        super().__init__()
        laplacian_kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.register_buffer("laplacian_kernel", laplacian_kernel, persistent=False)
        self.mask_bn = nn.BatchNorm2d(in_channels)
        self.processor_conv = Conv(in_channels, out_channels, 1, act=act)
        self.dwconv = DWConv(in_channels, in_channels, k=3, act=True) if use_dwconv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.laplacian_kernel.type_as(x)
        grad_map = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        grad_map = self.mask_bn(grad_map)
        grad_mask = torch.tanh(grad_map.abs())
        y = x * (1 + grad_mask)
        if self.dwconv is not None:
            y = self.dwconv(y)
        return self.processor_conv(y)


class E_Bottleneck(nn.Module):
    """Edge-enhanced bottleneck used inside C2f_E."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
        detail_use_dwconv: bool = True,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.detail_branch = LGDI(c1, c2, use_dwconv=detail_use_dwconv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x)) + self.detail_branch(x)
        return out + x if self.add else out


class C2fLGDI(nn.Module):
    """C2f with LGDI detail bottlenecks at P2."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        detail_use_dwconv: bool = True,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            E_Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0,
                detail_use_dwconv=detail_use_dwconv,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MultiBranchReparam(nn.Module):
    """Re-parameterizable convolution block with parallel conv and identity branches."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        g: int = 1,
        branches: Optional[List[Dict[str, Any]]] = None,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k_max = k
        self.stride = s
        self.groups = g
        self.deploy = deploy
        self.chk_id = c1 == c2 and s == 1

        if branches is None:
            branches = [{"type": "conv", "k": k}]
            if k > 1:
                branches.append({"type": "conv", "k": 1})
            if self.chk_id:
                branches.append({"type": "id"})

        if self.deploy:
            self.fused_conv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=True)
        else:
            self.branches = nn.ModuleList()
            self.branch_bns = nn.ModuleList()
            self.id_bn = None
            for spec in branches:
                btype = spec.get("type")
                if btype == "conv":
                    k_b = spec.get("k", self.k_max)
                    d_b = spec.get("d", 1)
                    self.branches.append(nn.Conv2d(c1, c2, k_b, s, autopad(k_b, d=d_b), d_b, groups=g, bias=False))
                    bn = nn.BatchNorm2d(c2)
                    if "bn_init" in spec:
                        nn.init.constant_(bn.weight, float(spec["bn_init"]))
                    self.branch_bns.append(bn)
                elif btype == "id" and self.chk_id:
                    self.id_bn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.fused_conv(x)
        out = 0.0
        for i, branch in enumerate(self.branches):
            out += self.branch_bns[i](branch(x))
        if self.id_bn is not None:
            out += self.id_bn(x)
        return out

    @staticmethod
    def _fuse_bn_tensor(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight
        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        std = (var + bn.eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return w * scale, beta - mean * gamma / std

    def _pad_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        if kernel.shape[-1] == self.k_max:
            return kernel
        pad = (self.k_max - kernel.shape[-1]) // 2
        return F.pad(kernel, (pad, pad, pad, pad))

    def _get_id_tensor(self, kernel_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c_out, c_in_g, _, _ = kernel_ref.shape
        device, dtype = kernel_ref.device, kernel_ref.dtype
        w_id = torch.zeros(c_out, c_in_g, self.k_max, self.k_max, device=device, dtype=dtype)
        idx = torch.arange(c_out, device=device)
        w_id[idx, idx % c_in_g, self.k_max // 2, self.k_max // 2] = 1.0
        return w_id, torch.zeros(c_out, device=device, dtype=dtype)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        """Fuse all branches into one convolution for deployment."""
        if self.deploy:
            return
        if self.training:
            raise RuntimeError("Call eval() before fusing frozen BatchNorm statistics.")
        ref_conv = self.branches[0]
        fused_w = torch.zeros(self.c2, self.c1 // self.groups, self.k_max, self.k_max, device=ref_conv.weight.device, dtype=ref_conv.weight.dtype)
        fused_b = torch.zeros(self.c2, device=ref_conv.weight.device, dtype=ref_conv.weight.dtype)
        for conv, bn in zip(self.branches, self.branch_bns):
            w, b = self._fuse_bn_tensor(conv, bn)
            fused_w += self._pad_kernel(w)
            fused_b += b
        if self.id_bn is not None:
            w_id, b_id = self._get_id_tensor(fused_w)
            fused_w += w_id
            fused_b += b_id
        self.fused_conv = nn.Conv2d(
            self.c1, self.c2, self.k_max, self.stride, autopad(self.k_max), groups=self.groups, bias=True
        ).to(device=fused_w.device, dtype=fused_w.dtype)
        self.fused_conv.weight.copy_(fused_w)
        self.fused_conv.bias.copy_(fused_b)
        del self.branches, self.branch_bns, self.id_bn
        self.deploy = True


class RepVitMlp(nn.Module):
    """RepViT-style pointwise channel mixer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        act_layer: Union[Callable[[], nn.Module], nn.Module, None] = nn.SiLU,
    ) -> None:
        super().__init__()
        if act_layer is None:
            act_layer = nn.SiLU
        self.act = act_layer() if isinstance(act_layer, type) else act_layer
        branches = [{"type": "conv", "k": 1}, {"type": "conv", "k": 1}]
        self.pw1 = MultiBranchReparam(in_dim, hidden_dim, k=1, branches=branches)
        self.pw2 = MultiBranchReparam(hidden_dim, in_dim, k=1, branches=branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw2(self.act(self.pw1(x)))

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        for module in (self.pw1, self.pw2):
            if hasattr(module, "switch_to_deploy") and not module.deploy:
                module.switch_to_deploy()


class SEBlock(nn.Module):
    """Squeeze-and-excitation block."""

    def __init__(self, c1: int, r: int = 4) -> None:
        super().__init__()
        c_hidden = max(c1 // r, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c1, c_hidden, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(c_hidden, c1, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.act(self.fc1(y))
        y = self.gate(self.fc2(y))
        return x * y


class RPMKRepViT(nn.Module):
    """RPMK-RepViT: independent 3x3/11x11 spatial stages, SE and residual FFN."""

    def __init__(
        self,
        c1: int,
        c2: int,
        e: float = 2.0,
        kernels: Optional[List[int]] = None,
        use_se: bool = True,
        use_mb1: bool = True,
    ) -> None:
        super().__init__()
        if use_mb1:
            mb1_branches = [{"type": "conv", "k": 3}, {"type": "conv", "k": 1}, {"type": "id"}]
            self.mb1 = MultiBranchReparam(c1, c1, k=3, g=c1, branches=mb1_branches)
        else:
            self.mb1 = nn.Identity()

        kernels = [1, 3, 5, 7, 9, 11] if kernels is None else kernels
        mb2_branches = [{"type": "conv", "k": k} for k in kernels] + [{"type": "id"}]
        k_max_2 = max(kernels)
        self.mb2 = MultiBranchReparam(c1, c1, k=k_max_2, g=c1, branches=mb2_branches)
        self.se = SEBlock(c1, r=4) if use_se else nn.Identity()
        self.channel_mixer = RepVitMlp(c1, int(c1 * e), act_layer=nn.SiLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mb1(x)
        y = self.mb2(y)
        y = self.se(y)
        identity = y
        y = self.channel_mixer(y)
        return identity + y

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        for module in self.children():
            fn = getattr(module, "switch_to_deploy", None)
            if callable(fn) and not getattr(module, "deploy", True):
                fn()


class C2fRPMK(nn.Module):
    """C2f with RPMK-RepViT context aggregation at P4."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        kernels: Optional[List[int]] = None,
        g: int = 1,
        e: float = 0.5,
        use_se: bool = True,
        use_mb1: bool = True,
    ) -> None:
        super().__init__()
        del shortcut, g
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            RPMKRepViT(self.c, self.c, e=2.0, kernels=kernels, use_se=use_se, use_mb1=use_mb1) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        for module in self.m:
            fn = getattr(module, "switch_to_deploy", None)
            if callable(fn):
                fn()


@torch.no_grad()
def fuse_all_reparam_modules(root: nn.Module) -> None:
    """Fuse all submodules that expose a ``switch_to_deploy`` hook."""
    for module in root.modules():
        fn = getattr(module, "switch_to_deploy", None)
        if callable(fn):
            fn()


class GhostModuleV2(nn.Module):
    """GhostNetV2 ghost module with optional DFC attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        s: int = 1,
        relu: bool = True,
        mode: str = "original",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.c2 = c2
        init_channels = math.ceil(c2 / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, init_channels, k, s, k // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        if self.mode == "attn":
            self.dfc_conv = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(1, 5), padding=(0, 2), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(5, 1), padding=(2, 0), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
            )
            self.gate_fn = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)[:, : self.c2]
        if self.mode == "attn":
            res = F.avg_pool2d(x, kernel_size=2, stride=2)
            res = self.gate_fn(self.dfc_conv(res))
            out = out * F.interpolate(res, size=out.shape[-2:], mode="nearest")
        return out


class GhostBottleneckV2(nn.Module):
    """GhostNetV2 bottleneck used for LDC-YOLO downsampling stages."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        e: float = 3.0,
        layer_id: int = 2,
        se_ratio: float = 0.25,
        args: Any = None,
    ) -> None:
        super().__init__()
        del args
        self.s = s
        c_mid = int(c2 * e)
        mode = "original" if layer_id <= 1 else "attn"
        self.ghost1 = GhostModuleV2(c1, c_mid, relu=True, mode=mode, dw_size=k)
        if self.s > 1:
            self.conv_dw = nn.Conv2d(c_mid, c_mid, k, stride=s, padding=autopad(k), groups=c_mid, bias=False)
            self.bn_dw = nn.BatchNorm2d(c_mid)
        self.se = SEBlock(c_mid, r=int(1 / se_ratio)) if se_ratio > 0 else nn.Identity()
        self.ghost2 = GhostModuleV2(c_mid, c2, relu=False, mode="original", dw_size=k)
        if c1 == c2 and self.s == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c1, k, stride=s, padding=autopad(k), groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c2, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ghost1(x)
        if self.s > 1:
            x = self.bn_dw(self.conv_dw(x))
        x = self.se(x)
        x = self.ghost2(x)
        return x + self.shortcut(residual)


class DCNv2(nn.Module):
    """Modulated deformable convolution using torchvision operators."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act: Union[bool, nn.Module] = True,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.offset_channels = 2 * kernel_size * kernel_size
        self.mask_channels = kernel_size * kernel_size
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            (self.offset_channels + self.mask_channels) * groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.constant_(self.offset_mask_conv.weight, 0.0)
        nn.init.constant_(self.offset_mask_conv.bias, 0.0)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0.0)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().contiguous()
        out = self.offset_mask_conv(x)
        offset, mask = torch.split(out, [self.offset_channels * self.groups, self.mask_channels * self.groups], dim=1)
        try:
            x = torchvision.ops.deform_conv2d(
                input=x,
                offset=offset.contiguous(),
                weight=self.weight.float(),
                bias=self.bias.float(),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=torch.sigmoid(mask).contiguous(),
            )
        except RuntimeError as err:
            raise RuntimeError(
                "DCNv2 requires compatible PyTorch/torchvision deform_conv2d operators."
            ) from err
        return self.act(x)


class VisDroneDCNDetect(Detect):
    """DCN-enhanced detection head for dense small-object detection."""

    def __init__(self, nc: int = 80, ch: Tuple[int, ...] = ()) -> None:
        super().__init__(nc, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), DCNv2(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), DCNv2(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )

    def forward(self, x):
        return super().forward([feature.contiguous() for feature in x])

    def bias_init(self) -> None:
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


# Legacy symbols remain available for existing YAMLs and checkpoints.
ConvStem = DLStem
_DetailBranch = LGDI
C2f_E = C2fLGDI
G_Bottleneck = RPMKRepViT
C2f_G = C2fRPMK
