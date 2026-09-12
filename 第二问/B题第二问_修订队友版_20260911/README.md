# B题第二问修订版队友分享包

## 先看什么

1. `第二问队友版.docx`：适合直接发给队友阅读；
2. `第二问论文正文.md`：可复制进整篇论文，公式保留LaTeX；
3. `第二问思路讲解.md`：快速理解模型、修改点和不能误写的结论；
4. `仿真结果摘要.md`：只看数值和解释边界；
5. `第二问审阅落实清单.md`：逐条记录本次审阅意见如何处理。

## 核心结果

首次测向完成后，相对首次读数方向向左或向右偏转35.0968度，沿直线移动1000米，再测一次。局部坐标为

\[
(a,b)=(818.1818,\ \pm574.9596)\ \mathrm m.
\]

这两个坐标是同一位移的分量，不是两段路。移动加第二次测向共205秒。

该点是“解析充分安全域＋首次中心线名义源＋一阶误差带＋最坏径向定位直径”模型下的闭式最优解。不要写成所有误差和后续清除任务下的无条件全局最优解。

## 文件说明

- `q2_theory.py`：安全条件、闭式解和坐标旋转；
- `q1_localization.py`：精确半平面交、退化处理、旋转卡壳和直径圆判定；
- `q2_validation.py`：确定性检验与20,000场离线随机验证；
- `test_question2.py`：19项回归测试；
- `generate_figures.py`：重新生成四组PNG和SVG图；
- `simulation_summary.json`：机器可读的汇总和确定性核验；
- `simulation_samples.csv`：20,000条离线样本明细；
- `fig1` 至 `fig4`：论文图像，PNG用于Word，SVG用于排版软件。

## 使用方法

在PowerShell中进入本文件夹，再执行以下命令。若电脑上的Python命令是 `py`，可把下面的 `python` 换成 `py`。

### 1 查看闭式点

```powershell
python q2_theory.py --x 0 --y 0 --bearing 0 --side both
```

把 `--x`、`--y` 换成首次检测点坐标，把 `--bearing` 换成首次读数即可获得全局坐标。

### 2 只复算四项确定性检验

```powershell
python q2_validation.py --deterministic-only
```

该命令不会覆盖随机样本文件。

### 3 运行回归测试

```powershell
python -m unittest -v test_question2.py
```

正确结果应为19项全部通过。

### 4 重新生成20,000条离线样本

```powershell
python q2_validation.py --scenarios 20000 --seed 20260921
```

该命令会覆盖 `simulation_samples.csv` 和 `simulation_summary.json`。如果只需要阅读现有结果，不必执行。

### 5 重新生成图像

```powershell
python generate_figures.py
```

需要安装 `numpy` 和 `matplotlib`，版本要求见 `requirements.txt`。

## 安全边界

这些命令都是独立的数学程序，不需要启动或连接Jammers Simulator。为了遵守当前阶段“不进行正式测试”的要求，请不要把本目录复制到官方模拟器插件目录，也不要执行任何对接模拟器的脚本；本分享包本身不包含模拟器控制代码。
