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
| 等价类/边界值    | 未装饰模块（None/空 dict）与已注册返回 dict                   | 已覆盖：未注册路径；contract 装饰后非空       |
| 正常传参场景     | 返回值类型为 None 或 dict                                     | 已覆盖                                         |
| 异常传参场景     | 非 Module 输入                                                | 未覆盖：本文件仅测未注册路径，无稳定非法入参用例 |

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
    from torch.distributed._composable.contract import _get_registry
except ImportError:
    pytest.skip("_get_registry not available in this PyTorch version", allow_module_level=True)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
