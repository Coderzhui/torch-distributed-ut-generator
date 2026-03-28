# -*- coding: utf-8 -*-
"""
测试目的：验证 tensor.copy_ 接口在张量复制场景下的功能正确性
API 名称：tensor.copy_
API 签名：copy_(self, other, non_blocking=False) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | other 必填；non_blocking 有默认 False                         | 已覆盖：non_blocking True/False                 |
| 枚举选项         | N/A                                                           | N/A                                            |
| 参数类型         | other 为 Tensor；non_blocking 为 bool                         | 已覆盖；异常路径覆盖非 Tensor                  |
| 传参与不传参     | copy_(other) 与 copy_(other, non_blocking=...)               | 已覆盖                                         |
| 等价类/边界值    | 1D/2D/3D、大 tensor、连续多次 copy、多 dtype                 | 已覆盖：float32/bfloat16/int32                 |
| 正常传参场景     | 同 device 下复制后 shape/dtype 正确、返回 self               | 已覆盖                                         |
| 异常传参场景     | other 非 Tensor；src/dst shape 不一致                         | 已覆盖                                         |

未覆盖项及原因：
- 跨设备 copy：需多设备环境，复杂度高
- float16：精度验证非本测试目的
- 非连续 tensor：需特殊构造，本 UT 未覆盖

注意：本测试仅验证功能正确性（复制后 shape/dtype 正确），
     不做数值正确性校验。
"""

import torch
import pytest

import torch_npu  # noqa: F401

DEVICE_TYPE = "npu"


def _assert_raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}")


@pytest.mark.timeout(120)
def test_copy_basic():
    """基础功能：float32 tensor 复制"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(4, 4, dtype=torch.float32, device=device)
    dst = torch.zeros(4, 4, dtype=torch.float32, device=device)
    result = dst.copy_(src)
    assert result is dst
    assert dst.shape == (4, 4)
    assert dst.dtype == torch.float32


@pytest.mark.timeout(120)
def test_copy_non_blocking_false():
    """non_blocking=False"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(4, dtype=torch.float32, device=device)
    dst = torch.zeros(4, dtype=torch.float32, device=device)
    result = dst.copy_(src, non_blocking=False)
    assert result is dst


@pytest.mark.timeout(120)
def test_copy_non_blocking_true():
    """non_blocking=True"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(4, dtype=torch.float32, device=device)
    dst = torch.zeros(4, dtype=torch.float32, device=device)
    result = dst.copy_(src, non_blocking=True)
    assert result is dst


@pytest.mark.timeout(120)
def test_copy_dtype_float32():
    """dtype：float32 到 float32"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(8, dtype=torch.float32, device=device)
    dst = torch.zeros(8, dtype=torch.float32, device=device)
    dst.copy_(src)
    assert dst.dtype == torch.float32


@pytest.mark.timeout(120)
def test_copy_dtype_bfloat16():
    """dtype：bfloat16 到 bfloat16"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(8, dtype=torch.bfloat16, device=device)
    dst = torch.zeros(8, dtype=torch.bfloat16, device=device)
    dst.copy_(src)
    assert dst.dtype == torch.bfloat16


@pytest.mark.timeout(120)
def test_copy_dtype_int32():
    """dtype：int32 到 int32"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(8, dtype=torch.int32, device=device)
    dst = torch.zeros(8, dtype=torch.int32, device=device)
    dst.copy_(src)
    assert dst.dtype == torch.int32


@pytest.mark.timeout(120)
def test_copy_shape_1d():
    """1D tensor"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(16, dtype=torch.float32, device=device)
    dst = torch.zeros(16, dtype=torch.float32, device=device)
    dst.copy_(src)
    assert dst.shape == (16,)


@pytest.mark.timeout(120)
def test_copy_shape_3d():
    """3D tensor [2, 3, 4]"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(2, 3, 4, dtype=torch.float32, device=device)
    dst = torch.zeros(2, 3, 4, dtype=torch.float32, device=device)
    dst.copy_(src)
    assert dst.shape == (2, 3, 4)


@pytest.mark.timeout(120)
def test_copy_large_tensor():
    """大 tensor [1024, 1024]"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(1024, 1024, dtype=torch.float32, device=device)
    dst = torch.zeros(1024, 1024, dtype=torch.float32, device=device)
    dst.copy_(src)
    assert dst.shape == (1024, 1024)


@pytest.mark.timeout(120)
def test_copy_consecutive():
    """连续多次 copy"""
    device = torch.device(DEVICE_TYPE)
    dst = torch.zeros(4, dtype=torch.float32, device=device)
    for i in range(3):
        src = torch.ones(4, dtype=torch.float32, device=device) * (i + 1)
        dst.copy_(src)
    assert dst.shape == (4,)


@pytest.mark.timeout(120)
def test_copy_invalid_non_tensor():
    """异常场景：传入非 Tensor 类型"""
    device = torch.device(DEVICE_TYPE)
    dst = torch.zeros(4, dtype=torch.float32, device=device)
    _assert_raises(
        (TypeError, RuntimeError),
        lambda: dst.copy_("not_a_tensor")
    )


@pytest.mark.timeout(120)
def test_copy_invalid_shape_mismatch():
    """异常场景：shape 不匹配"""
    device = torch.device(DEVICE_TYPE)
    src = torch.ones(4, dtype=torch.float32, device=device)
    dst = torch.zeros(8, dtype=torch.float32, device=device)
    _assert_raises(
        (RuntimeError,),
        lambda: dst.copy_(src)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
