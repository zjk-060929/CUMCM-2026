# B 题第一问：代码与论文插图

本文件夹实现以下计算流程：

\[
\text{示向度误差约束}
\rightarrow\text{半平面交}
\rightarrow\text{凸定位多边形}
\rightarrow\text{旋转卡壳求直径}
\rightarrow\text{直径圆覆盖检验}.
\]

## 文件说明

- `q1_localization.py`：第一问完整算法，可读取 JSON 数据并输出计算结果。
- `test_q1_localization.py`：核心几何算法的单元测试。
- `generate_figures.py`：生成全部论文插图。
- `example_input.json`：与论文计算示例对应的输入数据。
- `requirements.txt`：Python 依赖。
- `fig1_half_plane_intersection.*`：测向扇形及半平面交定位多边形。
- `fig2_rotating_calipers.*`：旋转卡壳与最远对踵点对。
- `fig3_diameter_circle_counterexample.*`：直径圆不能必然覆盖区域的反例。
- `fig4_algorithm_flowchart.*`：第一问完整求解流程图。

每幅图同时提供 PNG 和 SVG。PNG 可直接预览和插入论文；SVG 为矢量图，适合排版或进一步编辑。

## 运行环境

建议使用 Python 3.10 或更高版本。在本文件夹中执行：

```powershell
python -m pip install -r requirements.txt
```

## 运行示例

使用内置示例数据：

```powershell
python q1_localization.py
```

读取 JSON 文件：

```powershell
python q1_localization.py --input example_input.json
```

示例计算结果的主要数值为：

- 定位多边形顶点数：4；
- 定位区域直径：约 `49.385425824 m`；
- 直径圆半径：约 `24.692712912 m`；
- 该示例的直径圆能够覆盖定位多边形。

## 输入格式

```json
{
  "error_deg": 1.0,
  "measurements": [
    {"x": 0.0, "y": 0.0, "bearing_deg": 0.0},
    {"x": 1000.0, "y": 1000.0, "bearing_deg": 270.0}
  ]
}
```

其中：

- `x`、`y` 的单位为米；
- `bearing_deg` 为示向度，单位为度；
- 方位角从正东方向开始，按逆时针方向增加；
- `error_deg` 默认为题目规定的 `1.0` 度。

## 重新生成图片

```powershell
python generate_figures.py
```

## 运行测试

```powershell
python -m unittest test_q1_localization.py -v
```

测试包括正交测向交会、正方形半平面交、等边三角形反例、矩形直径、单次测向无界区域，以及旋转卡壳与暴力枚举的随机交叉验证。

## 建议的论文图题

1. 图 1：带有有界测角误差的交会定位区域示意图。
2. 图 2：基于旋转卡壳的凸多边形直径计算示意图。
3. 图 3：直径圆不能完全覆盖定位区域的反例。
4. 图 4：问题一定位区域直径及覆盖性判定流程图。
