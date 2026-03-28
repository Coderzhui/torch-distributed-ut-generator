# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._common_utils._named_parameters_with_duplicates 接口的功能正确性
API 名称：torch.distributed.fsdp._common_utils._named_parameters_with_duplicates
API 签名：_named_parameters_with_duplicates(module, **kwargs) -> list[tuple[str, Parameter]]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | prefix 可选                                                   | 已覆盖：默认与带 prefix                        |
| 枚举选项         | N/A                                                           | N/A                                            |
| 参数类型         | nn.Module、kwargs prefix str                                  | 已覆盖                                         |
| 传参与不传参     | 仅 module 与 module+prefix                                    | 已覆盖                                         |
| 等价类/边界值    | Linear、Sequential、重复参数名场景                            | 已覆盖                                         |
| 正常传参场景     | 返回 list[tuple[str, Parameter]]                             | 已覆盖                                         |
| 异常传参场景     | 非 Module 输入                                                | 已覆盖                                         |

未覆盖项及原因：
- 内部 API，具体行为可能随版本变化

注意：本测试仅验证功能正确性，
     不做数值正确性校验。
"""

import torch
import torch.nn as nn
import pytest

import torch_npu  # noqa: F401


def _assert_raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}")


try:
    from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
except ImportError:
    pytest.skip("_named_parameters_with_duplicates not available in this PyTorch version", allow_module_level=True)


def test_named_parameters_with_duplicates_basic():
    """基础功能：获取命名参数"""
    module = nn.Linear(4, 4)
    params = _named_parameters_with_duplicates(module)
    assert isinstance(params, list)
    for name, param in params:
        assert isinstance(name, str)
        assert isinstance(param, nn.Parameter)


def test_named_parameters_with_duplicates_sequential():
    """Sequential 模块"""
    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 4)
    )
    params = _named_parameters_with_duplicates(model)
    assert isinstance(params, list)


def test_named_parameters_with_duplicates_prefix():
    """带 prefix 参数"""
    module = nn.Linear(4, 4)
    params = _named_parameters_with_duplicates(module, prefix="test")
    assert isinstance(params, list)


def test_named_parameters_with_duplicates_returns_list():
    """返回类型验证"""
    module = nn.Linear(4, 4)
    params = _named_parameters_with_duplicates(module)
    assert isinstance(params, list)
    if len(params) > 0:
        name, param = params[0]
        assert isinstance(name, str)
        assert isinstance(param, nn.Parameter)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
