# B 题第二问最终成果包

本文件夹只包含确定采用的第二检测点方案，不包含旧参数或候选网格寻优内容。

核心结论为

\[
S_2=S_1+\frac{9000}{11}\boldsymbol u
\pm\frac{2000\sqrt{10}}{11}\boldsymbol v,
\]

即在以首次示向度为前向轴的局部坐标系中

\[
(a,|b|)=(818.1818,574.9596)\ \mathrm m,
\qquad \|S_2-S_1\|=1000\ \mathrm m.
\]

该点是“解析充分安全域内、一阶测向误差带模型、最坏径向定位直径”准则下的闭式最优解。点位由连续推导得到，不使用网格步长或经验系数。

## 文件说明

- `第二问论文正文.md`：可直接整理进论文的完整正文、公式和结果；
- `第二问思路讲解.md`：面向队友的简明推导说明；
- `仿真结果摘要.md`：仿真设置、统计量和有效性检查；
- `q2_theory.py`：安全候选域、闭式解和实际坐标转换；
- `q1_localization.py`：半平面交、旋转卡壳及直径圆覆盖检验；
- `q2_validation.py`：固定闭式点的 20 000 组蒙特卡洛验证及最小包围圆；
- `generate_figures.py`：重新生成全部正文图像；
- `test_question2.py`：数学结果和几何代码的回归测试；
- `simulation_summary.json`：仿真统计汇总；
- `simulation_samples.csv`：20 000 个场景的逐场景明细；
- `fig1_safe_region_and_optimum.*`：解析安全候选域与闭式点；
- `fig2_diameter_objective.*`：一元目标函数与精确定位区域；
- `fig3_simulation_validation.*`：固定点仿真结果；
- `requirements.txt`：Python 依赖。

每张图同时提供高分辨率 PNG 和可编辑 SVG。

## 复现方法

在本文件夹中执行：

```powershell
python q2_theory.py
python q2_validation.py --scenarios 20000 --seed 20260921
python generate_figures.py
python -m unittest -v test_question2.py
```

依赖为 Python 3.10 及以上版本、NumPy 和 Matplotlib：

```powershell
python -m pip install -r requirements.txt
```

## 已生成结果

- 解析名义最坏定位直径：$122.1855\ \mathrm m$；
- 名义最远场景的精确半平面交直径：$122.4626\ \mathrm m$；
- 20 000 组仿真定位直径均值：$49.3937\ \mathrm m$；
- 定位直径 90% 分位：$73.7318\ \mathrm m$；
- 第二点信号保留率：100%（20 000/20 000；安全性由解析证明保证）；
- 有效定位区域率与真实位置包含率：均为 100%；
- 最小包围圆半径不超过 $20\ \mathrm m$ 的比例：37.905%。

仿真仅作模型验证，不用于确定第二检测点坐标。
