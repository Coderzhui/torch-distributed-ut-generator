# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.device_mesh._get_device_handle 接口的功能正确性
API 名称：torch.distributed.device_mesh._get_device_handle
API 签名：_get_device_handle(device_type: str) -> Optional[module]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | device_type 为 str，无 None 语义；区分合法与非法字符串        | 已覆盖：非法 device_type；合法路径见正常场景    |
| 枚举选项         | N/A（非固定枚举字符串）                                       | N/A                                            |
| 参数类型         | device_type: str                                            | 已覆盖                                         |
| 传参与不传参     | 单必填参数 device_type                                        | 已覆盖：传入 "npu"                             |
| 等价类/边界值    | 合法 "npu" 与非法字符串                                       | 已覆盖                                         |
| 正常传参场景     | 合法输入下返回非 None 的 device 模块句柄                      | 已覆盖：_get_device_handle("npu")              |
| 异常传参场景     | 非法 device_type 返回 None                                    | 已覆盖：invalid_device                         |

未覆盖项及原因：
- 内部 API，行为可能随 PyTorch 版本变化；未对 torch 内部实现做版本矩阵测试

注意：本测试仅验证功能正确性，
     不做数值正确性校验。
"""

import torch
import pytest

import torch_npu  # noqa: F401

try:
    from torch.distributed.device_mesh import _get_device_handle
except ImportError:
    pytest.skip("_get_device_handle not available in this PyTorch version", allow_module_level=True)


def test_get_device_handle_supported_device():
    """获取 NPU device handle（本 UT 仅 NPU 环境）"""
    assert _get_device_handle("npu") is not None


def test_get_device_handle_invalid():
    """无效 device_type 返回 None"""
    result = _get_device_handle("invalid_device")
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
