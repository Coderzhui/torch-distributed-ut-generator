# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable.contract 装饰器的功能正确性
API 名称：torch.distributed._composable.contract
API 签名：contract(state_cls: type = _State) -> Callable

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | contract(state_cls=...) 默认 _State                           | 已覆盖：contract() 无参                        |
| 枚举选项         | N/A                                                           | N/A                                            |
| 参数类型         | state_cls 为 type                                            | 已覆盖：默认 _State                            |
| 传参与不传参     | contract() 与 contract(state_cls=...)                       | 已覆盖：无参调用                               |
| 等价类/边界值    | 返回可调用装饰器工厂                                          | 已覆盖                                         |
| 正常传参场景     | 可导入、可调用、返回装饰器函数                                | 已覆盖                                         |
| 异常传参场景     | N/A（未构造稳定非法 state_cls 负例）                         | 未覆盖：非法类型行为依赖版本                   |

未覆盖项及原因：
- 内部 API，行为可能随 PyTorch 版本变化
- 装饰器真实作用于模块并参与分布式：需完整多进程环境，本文件仅做可导入与工厂行为校验

注意：本测试仅验证功能正确性，
     不做数值正确性校验。
"""

import torch
import pytest

import torch_npu  # noqa: F401

try:
    from torch.distributed._composable import contract
    from torch.distributed._composable_state import _State
except ImportError:
    pytest.skip("contract not available in this PyTorch version", allow_module_level=True)


def test_contract_importable():
    """验证 contract 可导入"""
    assert contract is not None
    assert callable(contract)


def test_contract_returns_callable():
    """验证 contract 返回可调用对象"""
    result = contract()
    assert callable(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
