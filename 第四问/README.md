# 第四问交付入口

先阅读：[第四问建模算法与自建模拟结果](/Users/kkkw/Desktop/国赛/第四问建模算法与自建模拟结果.md)。

最终默认策略为905米边长、31点三角覆盖，结合顺路补测、主动定位、最小包围圆和有限光学清除。所有已提供成绩均为自建模拟结果，官方演练与三次正式测试尚未进行。

## 快速运行

在此文件夹中运行（Python 3.10及以上；Windows可将 `python3` 改为 `python`）：

```bash
python3 -m unittest -v test_q4
python3 run_q4.py --seed 42
```

默认完全离线，不连接任何模拟器。结果写入 `results/demo/`。请为需要保留的新运行指定不同的 `--output` 目录。

完整复现：

```bash
python3 experiments.py --random-cases 500 --stress-per-group 50
python3 spacing_study.py
python3 verify_http.py
```

绘图使用 `requirements.txt` 中的依赖，运行 `python3 plot_results.py`。核心策略和测试均不需要第三方包。

## 当前结果与归档

- `results/benchmark_summary.json`、`paired_cases.csv`、`stress_cases.csv` 是最终905米版本。
- `results/spacing_study*` 使用另一批100个场景比较间距。
- `results/sample_seed_*/` 为最终样例，日志与退出后真值分开存储。
- `results/http_smoke/` 为自建模拟器的真实本机HTTP验证。
- `results/archive_spacing990/` 为早期版本归档，不能与最终数字混用。
- `figures/` 提供4张PNG和4张SVG。

使用官方模拟器的方法及尚待完成的正式测试表，见完整说明。默认HTTP示例使用自建服务2027端口；官方通常是2026端口，机器人标识必须改为真实参赛队号。
