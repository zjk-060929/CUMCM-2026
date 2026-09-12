# B题第一问完整交接包：排序队列算法与特殊情况

版本：2026-09-12。建议先看本文件，再看 `第一问成果汇总_排序队列完整版.pdf` 或同名 Markdown。文件组织延续前一份队友包：主程序、测试、生成图件脚本、输入算例和成果汇总；新增特殊情况说明、独立校验、公式素材及传输校验清单。

## 在线阅读入口

- [完整报告 PDF](第一问成果汇总_排序队列完整版.pdf) · [Markdown 正文](第一问成果汇总_排序队列完整版.md)
- [算法通俗讲解](算法通俗讲解.md) · [公式推导与证明](公式推导补充.md)
- [修改说明](修改说明.md) · [验证说明](验证说明.md) · [图件与公式索引](图件与公式索引.md)
- [主求解器](q1_localization.py) · [算例输出](example_results.json) · [性能比较](results/benchmark.csv)

GitHub上的展开目录与同版本ZIP内容一致。独立下载整个包后，以上内部链接仍可使用。仓库下载入口见[项目首页](https://github.com/zjk-060929/CUMCM-2026)。

## 已完成什么

1. 将前面讨论的第二种方法落实为**按方向排序、分别维护上下边界队列**的完整算法。它属于排序增量半平面交的上下包络写法，与旧版单一闭合队列的内部结构不同。默认求解不调用第三种逐边界区间法，完整流程时间为 $O(m\log m)$，空间为 $O(m)$，其中 $m=2n$。
2. 补齐同向平行、反向平行、重复/冗余约束、空集、无界区域、单点、线段、角度跨界和浮点尺度问题；检测到不可可靠表达的交点时给出明确错误状态。
3. 对有界区域输出极点、直径、最远点对、直径圆及覆盖判定。证明：**直径等于区域直径的圆盘，不一定能覆盖整个区域**，并提供符合测向设定的数值反例。
4. 52项单元/回归测试通过。另有1500组独立一般半平面核验：217组有界、1030组空集、253组无界，分类无差异，有界直径最大差约 $1.42\times10^{-13}$ 米。原始记录和复现脚本均在包内。

这里的定位区域采用题目附录2的纯角度交会区域。未将目标圆域、接收圆盘裁进半平面交；未连接官方模拟器，包内结果均为自建算例。浮点边界的精度约定见成果报告第9节。

## 文件导航

| 文件或目录 | 用途 |
|---|---|
| `第一问成果汇总_排序队列完整版.pdf` | 可直接阅读、交给队友的完整报告，含模型、证明、算例、复杂度与实验 |
| `第一问成果汇总_排序队列完整版.md` | 与PDF共用内容源的可编辑正文 |
| `算法通俗讲解.md` | 从“地板、天花板、竖直截面”理解队列、求交和直径 |
| `公式推导补充.md` | 完整数学定义、正确性证明和边界分类推导 |
| `修改说明.md` | 相比旧版本的改动、特殊情况与剩余适用限制 |
| `验证说明.md` | 测试样本、独立校验、计时口径与结果解释 |
| `图件与公式索引.md` | 每张图和每个公式的用途、文件编号与报告图号 |
| `q1_localization.py` | 默认求解器，运行仅需NumPy |
| `test_legacy_regression.py` | 保留的21项旧回归检查，调用新版求解器 |
| `test_q1_special_cases.py` | 新增31项特殊情况与数值检查 |
| `run_checks.py` | 运行52项检查并保存日志和结果摘要 |
| `validate_independent.py` | 独立线性规划分类与交点枚举核验 |
| `benchmark.py` | 同机、同输入、交替计时的三实现比较 |
| `generate_figures.py` | 重建9幅PNG/SVG图件、10个公式的PNG/SVG/TeX和算例输出 |
| `report_content.py`、`build_report.py` | 报告内容源和Markdown/PDF生成器 |
| `example_input.json` | 常规270°交会，直径圆能覆盖 |
| `counterexample_input.json` | 269°交会反例，直径圆不能覆盖 |
| `point_input.json`、`segment_input.json` | 测向角形交集退化成单点、线段 |
| `examples/` | 空集、无界象限、条带、极薄三角形输入 |
| `example_results.json` | 上述8组输入对应的输出 |
| `figures/` | 9幅图片，每幅同时提供PNG和SVG |
| `formulas/` | 10个公式的PNG、SVG，以及可编辑LaTeX和JSON源码 |
| `results/` | 测试日志、独立验证结果、计时CSV与环境记录 |
| `reference/` | 两份旧算法的原样副本，只用于比较，不是默认运行依赖 |
| `题目依据/B题.pdf` | 题目原件副本 |
| `version_manifest.json` | 本版范围、来源与验证环境 |
| `文件清单_SHA256.csv`、`check_package.py` | 包内文件大小与SHA256校验 |

## 最快开始

先完整解压，不要只复制主程序之外的某一张图片。打开本包目录后运行：

```powershell
python -m pip install numpy
python q1_localization.py --input example_input.json
python q1_localization.py --input counterexample_input.json
python q1_localization.py --input point_input.json
python q1_localization.py --input segment_input.json
python run_checks.py
```

建议Python 3.10或以上。已在Windows 11、Python 3.12.14、NumPy 2.3.5下验证。项目不包含Python安装程序和第三方依赖安装包。

| 算例 | 输出类型 | 直径/米 | 直径圆覆盖 |
|---|---|---:|---|
| `example_input.json` | polygon | 49.385425824 | 是 |
| `counterexample_input.json` | polygon | 49.362859759 | 否 |
| `point_input.json` | point | 0 | 是 |
| `segment_input.json` | segment | 3.491012986 | 是 |

反例圆半径24.681429880米，最远顶点距圆心25.115941025米，超出0.434511145米。结果使用完整浮点数计算，再将JSON展示值舍入到9位小数。

## 输入与输出

测向输入示例：角度从正东逆时针计量，坐标单位米，半角默认1°。

```json
{
  "error_deg": 1.0,
  "measurements": [
    {"x": 0, "y": 0, "bearing_deg": 0},
    {"x": 1000, "y": 1000, "bearing_deg": 269}
  ]
}
```

也支持一般半平面入口。以下输入是 $x\ge0,x\le4,y\ge0,y\le3$：

```json
{"halfplanes": [[1, 0, 0], [-1, 0, -4], [0, 1, 0], [0, -1, -3]]}
```

每条 `[a,b,c]` 表示 $ax+by\ge c$。两类输入二选一。`halfplanes: []` 表示无有效约束、整个平面无界；空测向记录则作为无意义测向输入拒绝。

| 状态 | 含义 | 退出码 |
|---|---|---:|
| `polygon` | 正面积有界凸多边形 | 0 |
| `segment` | 有界线段 | 0 |
| `point` | 单点，直径0 | 0 |
| `empty` | 约束矛盾，不给出有限直径 | 2 |
| `unbounded` | 无界，不给出有限直径 | 2 |
| `numerical_error` | 交点溢出或数值复核失败 | 2 |
| `invalid_input` | 格式、数值或参数无效 | 2 |

非零退出码在这里也用于可预期的几何状态。不能将“没有输出多边形”一概解释为算法崩溃，也不能将无界区域报成直径0。

Python调用接口：

```python
from q1_localization import Measurement, solve_localization, LocalizationError

try:
    result = solve_localization([
        Measurement(0, 0, 0),
        Measurement(1000, 1000, 269),
    ])
    print(result.region_type, result.diameter, result.circle_covers_polygon)
except LocalizationError as error:
    print(error.status, str(error))
```

## 重建全部材料

先校验收到的原包：

```powershell
python check_package.py
```

依次运行，计时期间避免同时运行其他计算任务：

```powershell
python -m pip install -r requirements-validation.txt
python -m pip install -r requirements-report.txt
python run_checks.py
python validate_independent.py --cases 1500
python benchmark.py
python generate_figures.py
python build_report.py
```

`benchmark.py`只计本机几何计算，不含HTTP、文件输出或实际检测。低阶复杂度不保证小输入最快。旧闭合队列参考代码的最终逐点逐约束复核本身可能达到二次复杂度；不能把它的实测曲线等同于理想排序队列算法。

图片生成需要可用中文字体；PDF默认尝试Windows宋体及常见macOS/Linux中文TrueType字体。缺字体时通过环境变量 `Q1_REPORT_FONT` 指定中文TTF/TTC。SVG中的字体已经转为路径，阅读现成图件不需要安装相同字体；报告PDF嵌入中文字体。

重建会更新实验耗时、图件、报告等文件，原交付SHA256清单随之失效，这是正常现象。要保留原始可核验记录，请先复制整个包。标准LaTeX公式文件可单独编辑或编译；包内PNG/SVG可直接插入论文或Word。

## 交接时需要保留的限定

- 新版采用排序队列的上下包络形式。第三种逐边界区间法只在 `reference/` 作比较。
- 支持一般**线性半平面**，不直接支持圆弧边界、非凸区域或多个干扰源的观测关联。
- 点/线段在本包是合法几何输出；实际测向是否可能产生这些极端数据，属于数据来源问题。
- 数值测试支持本版在已覆盖尺度下使用，不构成任意浮点输入的精确保证。
- 直径圆和最小包围圆不同；本包默认求解器不计算通用最小包围圆。
- 本次独立交付第一问；没有覆盖第二问第三版、第三问或队友原始工作目录。

排序增量方法的背景可参阅 [cp-algorithms的半平面交说明](https://cp-algorithms.com/geometry/halfplane-intersection.html)。本版上下包络实现的证明和数值约定以包内文档为准。
