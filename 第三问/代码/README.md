# 离线代码使用说明

## 文件作用

- `q3_iteration_runner.py`：十轮候选策略统一离线运行器，第9轮名称为 `joint_scale125`；
- `q3_orienteering_candidate.py`：第9轮联合滚动有向弧策略；
- `geometry.py`：七点覆盖、第二观测点、半平面交、最小包围圆和清除网格；
- `policy.py`：历史保证性检测、定位与清除流程；
- `offline_validation.py`：旧坐标哈希误差场景生成器；
- `scratch_q3_global_rolling.py`：记录时间分解的离线客户端；
- `scheduler.py`、`q3_fast_scheduler.py`：状态压缩路线规划及Numba加速；
- `protocol.py`：仅用于满足历史策略类型依赖；本包没有实时运行入口。

原实验脚本含本机绝对路径，本包只把导入路径改为相对当前文件夹，算法和场景生成逻辑未改动。

## 环境

建议使用Python 3.11，并安装：

```powershell
python -m pip install numpy numba
```

## 三场快速复核

在本文件夹打开PowerShell：

```powershell
python q3_iteration_runner.py --variant joint_scale125 --scenarios 3 --seed 20260913 --workers 1 --output "..\数据\复现冒烟_3场_新运行.json"
```

本包已经执行过一次三场复核，结果保存在 `数据/复现冒烟_3场.json`。三个场景的总时间和移动距离均与原500场文件中编号0、1、2完全一致。

## 完整500场复现

```powershell
python q3_iteration_runner.py --variant joint_scale125 --scenarios 500 --seed 20260913 --workers 8 --output "..\数据\复现_round09_500.json"
```

`workers` 可按电脑核心数调整，只影响本地运行速度，不影响虚拟时间。完整运行较慢；若只是检查文件是否可用，三场复核即可。

预期完整汇总的关键值为：

```text
successes = 500
mean_total_s = 3211.3899771214974
mean_per_source_s = 250.85859419926635
p90_total_s = 3589.0197279256304
```

## 安全说明

该运行器只创建内存中的离线客户端，不打开端口、不访问网络，也不调用模拟器 `/enter`。请勿把本包误改成实时或正式测试程序。
