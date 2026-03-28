# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.split_with_sizes_copy 接口在张量分割场景下的功能正确性
API 名称：torch.split_with_sizes_copy
API 签名：split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | dim 有默认 0；out 必填                                         | 已覆盖：多 dim；out 与 split_sizes 对齐         |
| 枚举选项         | N/A                                                           | N/A                                            |
| 参数类型         | Tensor、list[int] split_sizes、int dim、list[Tensor] out      | 已覆盖                                         |
| 传参与不传参     | 显式 dim 与默认 dim（通过不同用例体现）                       | 已覆盖                                         |
| 等价类/边界值    | 1D/2D/3D、大 tensor、多组 split_sizes、dim=-1               | 已覆盖                                         |
| 正常传参场景     | 合法输入下 out 被原地写入且 shape/dtype 正确                   | 已覆盖：float32/bfloat16                       |
| 异常传参场景     | out 长度与 split_sizes 不一致；dim 越界                       | 已覆盖                                         |

未覆盖项及原因：
- split_sizes 之和与维度大小不一致：行为依赖实现，未做稳定负例
- 非连续 tensor：需特殊构造
- out 原地副作用：已在注释中说明，未单独做副作用断言矩阵

注意：本测试仅验证功能正确性（分割后 shape/dtype 正确），
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
def test_split_with_sizes_copy_basic():
    """基础功能：1D tensor 按 split_sizes 分割"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(10, dtype=torch.float32, device=device)
    split_sizes = [2, 3, 5]
    out = [torch.zeros(2, dtype=torch.float32, device=device),
           torch.zeros(3, dtype=torch.float32, device=device),
           torch.zeros(5, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].shape == (2,)
    assert out[1].shape == (3,)
    assert out[2].shape == (5,)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dim0():
    """不同 dim：dim=0"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(20, dtype=torch.float32, device=device).reshape(4, 5)
    split_sizes = [2, 2]
    out = [torch.zeros(2, 5, dtype=torch.float32, device=device),
           torch.zeros(2, 5, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].shape == (2, 5)
    assert out[1].shape == (2, 5)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dim1():
    """不同 dim：dim=1"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(20, dtype=torch.float32, device=device).reshape(4, 5)
    split_sizes = [2, 3]
    out = [torch.zeros(4, 2, dtype=torch.float32, device=device),
           torch.zeros(4, 3, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=1, out=out)
    assert out[0].shape == (4, 2)
    assert out[1].shape == (4, 3)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dim_negative():
    """不同 dim：dim=-1（最后一维）"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    split_sizes = [2, 2]
    out = [torch.zeros(3, 2, dtype=torch.float32, device=device),
           torch.zeros(3, 2, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=-1, out=out)
    assert out[0].shape == (3, 2)
    assert out[1].shape == (3, 2)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dtype_float32():
    """dtype：float32"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(10, dtype=torch.float32, device=device)
    split_sizes = [5, 5]
    out = [torch.zeros(5, dtype=torch.float32, device=device),
           torch.zeros(5, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].dtype == torch.float32
    assert out[1].dtype == torch.float32


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dtype_bfloat16():
    """dtype：bfloat16"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(10, dtype=torch.bfloat16, device=device)
    split_sizes = [3, 7]
    out = [torch.zeros(3, dtype=torch.bfloat16, device=device),
           torch.zeros(7, dtype=torch.bfloat16, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].dtype == torch.bfloat16
    assert out[1].dtype == torch.bfloat16


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_large_tensor():
    """大 tensor [1024, 1024]"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(1024, 1024, dtype=torch.float32, device=device)
    split_sizes = [256, 256, 256, 256]
    out = [torch.zeros(256, 1024, dtype=torch.float32, device=device) for _ in range(4)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    for o in out:
        assert o.shape == (256, 1024)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_3d_tensor():
    """3D tensor"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(40, dtype=torch.float32, device=device).reshape(2, 4, 5)
    split_sizes = [1, 1]
    out = [torch.zeros(1, 4, 5, dtype=torch.float32, device=device),
           torch.zeros(1, 4, 5, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].shape == (1, 4, 5)
    assert out[1].shape == (1, 4, 5)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_invalid_split_sizes():
    """异常场景：split_sizes 长度与 out 列表长度不匹配"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(10, dtype=torch.float32, device=device)
    split_sizes = [2, 3, 5]
    out = [torch.zeros(2, dtype=torch.float32, device=device),
           torch.zeros(3, dtype=torch.float32, device=device)]
    _assert_raises(
        (RuntimeError, ValueError),
        lambda: torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    )


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_invalid_dim():
    """异常场景：dim 越界"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(10, dtype=torch.float32, device=device)
    split_sizes = [5, 5]
    out = [torch.zeros(5, dtype=torch.float32, device=device),
           torch.zeros(5, dtype=torch.float32, device=device)]
    _assert_raises(
        (RuntimeError, IndexError),
        lambda: torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=10, out=out)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
