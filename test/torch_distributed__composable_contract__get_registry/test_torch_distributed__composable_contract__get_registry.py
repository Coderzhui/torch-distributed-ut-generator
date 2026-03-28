# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable.contract._get_registry 接口的功能正确性
API 名称：torch.distributed._composable.contract._get_registry
API 签名：_get_registry(module) -> Optional[dict]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | N/A                                                           | N/A                                            |
| 枚举选项         | N/A                                                           | N/A                                            |
| 参数类型         | nn.Module                                                    | 已覆盖                                         |
| 传参与不传参     | 单参 module                                                   | 已覆盖                                         |
| 等价类/边界值    | 未装饰模块与经 @contract API 调用后的已注册模块               | 已覆盖：未注册 None；调用后非 None dict       |
| 正常传参场景     | 返回值类型为 None 或 dict                                     | 已覆盖                                         |
| 异常传参场景     | 非 Module 输入                                                | 未覆盖：无稳定非法入参用例                     |

未覆盖项及原因：
- 内部 API，具体行为可能随版本变化
- 非 nn.Module 入参：未写负例，避免依赖未文档化行为

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
    from torch.distributed._composable.contract import _get_registry, contract
except ImportError:
    pytest.skip("_get_registry not available in this PyTorch version", allow_module_level=True)


@contract()
def _ut_registry_probe(module: nn.Module) -> nn.Module:
    """仅用于 UT：对 module 应用一次 composable contract，以写入 REGISTRY。"""
    return module


def test_get_registry_basic():
    """基础功能：获取模块注册表，未注册返回 None"""
    module = nn.Linear(4, 4)
    registry = _get_registry(module)
    assert registry is None or isinstance(registry, dict)


def test_get_registry_unregistered():
    """未注册的模块返回 None"""
    module = nn.Linear(4, 4)
    result = _get_registry(module)
    assert result is None


def test_get_registry_registered():
    """已调用 @contract 装饰 API 的模块：注册表非 None，且包含该 API 名称"""
    module = nn.Linear(4, 4)
    assert _get_registry(module) is None
    _ut_registry_probe(module)
    registry = _get_registry(module)
    assert registry is not None
    assert isinstance(registry, dict)
    assert "_ut_registry_probe" in registry


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
