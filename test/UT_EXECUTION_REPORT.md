# UT 执行结果报告（最新）

## 执行信息

- 执行日期: 2026-03-28
- 工作目录: `/home/xxxxxxxx/pta_ut`
- 解释器: `/usr/local/python311/bin/python3.11`（Python 3.11.9）
- 执行方式: **串行**（pytest 默认单进程顺序执行，未使用 `pytest-xdist` 等并行插件）
- 命令:
  ```bash
  /usr/local/python311/bin/python3.11 -m pytest test -v --tb=short -rs
  ```
- 完整终端输出已保存: `test/UT_EXECUTION_REPORT_FULL_RUN.log`

## 结果总览

| 项 | 数量 |
|----|------|
| 收集用例 | 92 |
| 通过 | 92 |
| 失败 | 0 |
| 跳过 | 0 |
| 告警 | 28 |
| 总耗时 | 345.94 s（约 5 分 46 秒） |
| 退出码 | 0 |

- 通过率: **100%**（92/92）

## 跳过用例（`-rs`）

- 无：本次运行未产生 skip。

## 告警摘要

- **`PytestUnknownMarkWarning: Unknown pytest.mark.timeout`**  
  环境未注册 `pytest-timeout` 的 `timeout` mark，涉及文件包括：
  - `test/tensor_copy_/test_tensor_copy_.py`
  - `test/torch_distributed__composable_state__insert_module_state/test_torch_distributed__composable_state__insert_module_state.py`
  - `test/torch_split_with_sizes_copy/test_torch_split_with_sizes_copy.py`  
  如需消除告警：安装 `pytest-timeout` 或在 `pytest.ini` / `pyproject.toml` 中注册该 mark。

## pytest 汇总行

```
92 passed, 28 warnings in 345.94s (0:05:45)
```
