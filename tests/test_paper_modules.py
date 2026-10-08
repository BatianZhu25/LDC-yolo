import copy
import torch
import pytest
from ldc_yolo import LGDI, RPMKRepViT, fuse_all_reparam_modules
from ldc_yolo.loss import wiou_v1


def test_lgdi_stencil_and_bounded_modulation():
    layer = LGDI(4, 4).eval()
    assert layer.laplacian_kernel[0, 0, 1, 1] == 8
    assert layer.laplacian_kernel[0].sum() == 0
    x = torch.randn(2, 4, 12, 12)
    seen = []
    handle = layer.dwconv.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0]))
    with torch.no_grad():
        result = layer(x)
    handle.remove()
    ratio = seen[0] / x
    assert torch.all(ratio >= 1) and torch.all(ratio <= 2)
    assert result.shape == x.shape and torch.isfinite(result).all()


def test_rpmk_stagewise_fusion_with_frozen_bn():
    block = RPMKRepViT(8, 8).eval().double()
    for module in block.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.3, 1.5)
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.uniform_(-0.3, 0.3)
    x = torch.randn(2, 8, 16, 16, dtype=torch.float64)
    with torch.no_grad():
        expected = block(x)
        fused = copy.deepcopy(block)
        fuse_all_reparam_modules(fused)
        actual = fused(x)
    assert fused.mb1.fused_conv.kernel_size == (3, 3)
    assert fused.mb2.fused_conv.kernel_size == (11, 11)
    assert fused.mb1.fused_conv.groups == fused.mb2.fused_conv.groups == 8
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


def test_fusion_rejects_training_mode():
    with pytest.raises(RuntimeError, match="eval"):
        fuse_all_reparam_modules(RPMKRepViT(4, 4))


def test_wiou_identical_boxes_and_gradient():
    target = torch.tensor([[0., 0., 2., 2.]])
    torch.testing.assert_close(wiou_v1(target, target), torch.zeros(1, 1), atol=1e-6, rtol=0)
    pred = torch.tensor([[0.2, 0.1, 1.8, 1.9]], requires_grad=True)
    loss = wiou_v1(pred, target).sum()
    loss.backward()
    assert loss > 0 and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
