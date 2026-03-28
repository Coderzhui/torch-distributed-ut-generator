# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work.wait 方法在不同场景下的功能正确性
API 名称：torch.distributed.Work.wait
API 签名：wait(self, timeout=...) -> bool

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | wait(timeout=None) 与带超时（若用例使用）                    | 已覆盖：默认 wait；多次 wait                   |
| 枚举选项         | 集合类型 all_reduce / broadcast / all_gather                  | 已覆盖                                         |
| 参数类型         | Work.wait 返回值 bool 或 None；Tensor dtype                   | 已覆盖：float32/bfloat16                       |
| 传参与不传参     | 异步 op 返回 Work 后调用 wait                                 | 已覆盖                                         |
| 等价类/边界值    | 大 tensor、自定义子组、连续多次 wait                          | 已覆盖                                         |
| 正常传参场景     | wait 返回 True/None，集合完成后 tensor 维度保持               | 已覆盖                                         |
| 异常传参场景     | N/A（未构造超时失败、后端错误等稳定负例）                     | 未覆盖：精确超时与后端异常模拟                 |

未覆盖项及原因：
- 超时失败路径：需精确时序，环境难以稳定复现
- 后端异常：模拟成本高
- get_future()：与后端能力相关，非本文件重点

注意：本测试仅验证功能正确性（wait 返回正确、状态一致），
     不做数值正确性校验。
"""

import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

import torch_npu  # noqa: F401

DEVICE_TYPE = "npu"
BACKEND = "hccl"
WORLD_SIZE = 2


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def _setup_device(rank):
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch.npu.set_device(rank)


def _worker(rank, world_size, port, test_name):
    _setup_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)
    try:
        if test_name == "test_wait_all_reduce":
            _test_wait_all_reduce(rank, world_size)
        elif test_name == "test_wait_broadcast":
            _test_wait_broadcast(rank, world_size)
        elif test_name == "test_wait_all_gather":
            _test_wait_all_gather(rank, world_size)
        elif test_name == "test_wait_multiple":
            _test_wait_multiple(rank, world_size)
        elif test_name == "test_wait_dtype_float32":
            _test_wait_dtype_float32(rank, world_size)
        elif test_name == "test_wait_dtype_bfloat16":
            _test_wait_dtype_bfloat16(rank, world_size)
        elif test_name == "test_wait_large_tensor":
            _test_wait_large_tensor(rank, world_size)
        elif test_name == "test_wait_custom_group":
            _test_wait_custom_group(rank, world_size)
    finally:
        dist.destroy_process_group()


def _run_test(test_name):
    port = _get_free_port()
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, port, test_name),
        nprocs=WORLD_SIZE,
        join=True,
    )


def _test_wait_all_reduce(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    result = work.wait()
    assert result is True or result is None
    assert tensor.shape == (4, 4)


def _test_wait_broadcast(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.broadcast(tensor, src=0, async_op=True)
    result = work.wait()
    assert result is True or result is None
    assert tensor.shape == (4, 4)


def _test_wait_all_gather(rank, world_size):
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    tensor_list = [torch.zeros(4, dtype=torch.float32, device=DEVICE_TYPE) for _ in range(world_size)]
    work = dist.all_gather(tensor_list, tensor, async_op=True)
    result = work.wait()
    assert result is True or result is None
    for t in tensor_list:
        assert t.shape == (4,)


def _test_wait_multiple(rank, world_size):
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    result1 = work.wait()
    result2 = work.wait()
    assert result1 is True or result1 is None
    assert result2 is True or result2 is None


def _test_wait_dtype_float32(rank, world_size):
    tensor = torch.ones(8, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    assert tensor.dtype == torch.float32


def _test_wait_dtype_bfloat16(rank, world_size):
    tensor = torch.ones(8, dtype=torch.bfloat16, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    assert tensor.dtype == torch.bfloat16


def _test_wait_large_tensor(rank, world_size):
    tensor = torch.ones(1024, 1024, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    assert tensor.shape == (1024, 1024)


def _test_wait_custom_group(rank, world_size):
    sub_group = dist.new_group(ranks=list(range(world_size)))
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, group=sub_group, async_op=True)
    result = work.wait()
    assert result is True or result is None


def test_wait_all_reduce():
    _run_test("test_wait_all_reduce")


def test_wait_broadcast():
    _run_test("test_wait_broadcast")


def test_wait_all_gather():
    _run_test("test_wait_all_gather")


def test_wait_multiple():
    _run_test("test_wait_multiple")


def test_wait_dtype_float32():
    _run_test("test_wait_dtype_float32")


def test_wait_dtype_bfloat16():
    _run_test("test_wait_dtype_bfloat16")


def test_wait_large_tensor():
    _run_test("test_wait_large_tensor")


def test_wait_custom_group():
    _run_test("test_wait_custom_group")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
