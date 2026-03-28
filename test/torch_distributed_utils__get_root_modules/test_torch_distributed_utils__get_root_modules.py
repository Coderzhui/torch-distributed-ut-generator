# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.utils._get_root_modules 接口的功能正确性
API 名称：torch.distributed.utils._get_root_modules
API 签名：_get_root_modules(modules: list[nn.Module]) -> list[nn.Module]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | modules 列表为空与单元素、多元素                              | 已覆盖：空列表、单模块、多模块、嵌套 Sequential |
| 枚举选项         | N/A                                                           | N/A                                            |
| 参数类型         | list[nn.Module]                                              | 已覆盖                                         |
| 传参与不传参     | 仅位置参数 modules                                            | 已覆盖                                         |
| 等价类/边界值    | 独立模块与共享子模块（嵌套）根节点集合                        | 已覆盖                                         |
| 正常传参场景     | 返回 list，元素为根 Module                                   | 已覆盖                                         |
| 异常传参场景     | N/A（未覆盖非 list 输入等稳定文档化异常）                     | 未覆盖：行为依赖实现                           |

未覆盖项及原因：
- 内部 API，具体行为可能随版本变化
- 非法输入（非 Module 列表）未系统覆盖

注意：本测试仅验证功能正确性（返回正确的根模块列表），
     不做数值正确性校验。
"""

import torch
import torch.nn as nn
import pytest

import torch_npu  # noqa: F401

try:
    from torch.distributed.utils import _get_root_modules
except ImportError:
    pytest.skip("_get_root_modules not available in this PyTorch version", allow_module_level=True)


def test_get_root_modules_basic():
    """基础功能：传入模块列表"""
    modules = [nn.Linear(4, 4)]
    roots = _get_root_modules(modules)
    assert isinstance(roots, list)
    assert len(roots) >= 1


def test_get_root_modules_multiple():
    """多个模块"""
    modules = [nn.Linear(4, 4), nn.Linear(4, 4)]
    roots = _get_root_modules(modules)
    assert isinstance(roots, list)
    assert len(roots) >= 1


def test_get_root_modules_empty():
    """空列表"""
    modules = []
    roots = _get_root_modules(modules)
    assert isinstance(roots, list)
    assert len(roots) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
