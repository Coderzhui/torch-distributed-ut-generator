## **路径命名**：覆盖主技能的规定（去掉前缀 `torch.distributed` 再转下划线）。例：`torch.distributed.all_reduce` → **`_all_reduce`**。

---

## 多卡用例是否需要（分布式 API 优先生效）

生成或增补 **`torch.distributed`** 相关 UT 时，**先判断**该 API 的语义是否**必须**在多进程、多设备（`world_size ≥ 2` 或多 NPU）下才能覆盖；**不要**默认给每个分布式 API 都写多卡用例。

**通常不需要专门多卡 NPU 用例（单进程 / `world_size=1` / CPU+gloo 或单卡 NPU 即可）**

- 纯 Python 工具、数据结构、装饰器状态机：如 `utils._get_root_modules`、`_composable.contract` / `_get_registry`、`_composable_state._insert_module_state`、仅构造 `TensorMeta` 等。
- 文档或实现明确在单 rank 下行为已完整：例如仅依赖「已初始化进程组」但 rank 维度无分支。
- `Work` / `Work.wait` 等：若仅需验证「异步返回类型 + `wait` 不报错」，可用 **gloo + `world_size=1`** 作基线；多卡仅在有 **rank 互斥语义** 或 **后端 HCCL 多卡路径** 必须覆盖时再加。

**需要多卡 NPU 用例（应写 `mp.spawn` 等，并对测试方法使用 `@skipIfUnsupportMultiNPU(n)`）**

- **集合通信 / P2P** 且语义依赖 **多 rank**：`broadcast` 的 src/dst、`all_gather` 拼 shape、`send`/`recv`、`new_group` 子组等。
- **显式要求多设备协同**：如 `can_device_access_peer(0,1)`、多 rank `DeviceMesh`、`DTensor` **Shard** 跨 rank 一致性等。
- **ascend_pytorch/test/distributed/** 或 **pytorch/test/distributed/** 中同类 API **始终用多进程多卡** 范本时，应对齐该范式。

**落地要求**

- 在 UT 文件头部 **覆盖维度表** 或 **未覆盖项** 中简要写明：本 API **是否**含多卡用例、若不含则说明原因（例如「单 rank 可验证全部参数面」或「多卡行为与 ascend 某文件一致，由 xxx 覆盖」）。
- 若用户或场景明确要求验证 **HCCL 多卡**，再增加多卡用例，避免无意义的多卡用例拖慢 CI、或在单卡环境大量 skip。

---

## TestCase 与 run_tests 模板（推荐）

```python
import unittest
import torch

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    torch_npu = None

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

_DEVICE_TYPE = torch._C._get_privateuse1_backend_name()


class TestTorchXxxApi(TestCase):
    def test_example_on_privateuse1(self):
        # 使用 torch.device(_DEVICE_TYPE) 或 ascend 惯用写法放置张量
        ...


if __name__ == "__main__":
    run_tests()
```

多卡示例：

```python
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestTorchXxxMulti(TestCase):
    @skipIfUnsupportMultiNPU(2)
    def test_two_devices(self):
        ...
```

## 自检清单

- [ ] 路径与文件名无 `.`，且已按本文「路径命名」（去掉前缀 `torch.distributed` 再转下划线）
- [ ] 已按上文 **「多卡用例是否需要」** 做过判断；多卡用例仅在有语义必要时编写，并在 docstring 中可追溯到理由
- [ ] 仅改动 `test/`；头部中文 docstring 与覆盖表已按实填写
- [ ] unittest + TestCase；无 pytest；无数值精度断言
- [ ] NPU 用例占比符合要求；`device` 使用 `_get_privateuse1_backend_name()`
- [ ] 凡 **需要 ≥2 卡** 的测试方法均带 `@skipIfUnsupportMultiNPU(n)`
- [ ] NPU 环境执行后已写 `UT_REPORT.md` 或 `UT_EXECUTION_REPORT.md`
