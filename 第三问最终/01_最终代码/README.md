# B题第三问 Jammer 无门禁代码

本副本已经移除以下运行门禁：

- `allow_live_practice` 配置开关；
- `--live-practice` 命令行开关；
- 人工核对清单和随机确认短语；
- 运行前自动把配置恢复为 `false` 的逻辑。

队伍编号已经写入 `config.json`：`202617241078`，接口固定为本机
`127.0.0.1:2026`。程序不会启动或点击模拟器界面。

## 运行

先在 Jammer 中打开本次测试并等待页面显示“等待机器狗进入”，抄下实际案例编码。
然后在本文件夹打开 PowerShell。假设案例编码为
`AB12-CD34-EF56-GH78`，运行：

```powershell
& "D:\Anaconda\python.exe" run_practice.py --case-code "AB12-CD34-EF56-GH78"
```

命令执行后将直接访问 `/enter`，不会再要求确认。案例编码只用于本地日志，
不会发送给模拟器。

只想离线查看配置而不连接时运行：

```powershell
& "D:\Anaconda\python.exe" run_practice.py --case-code "AB12-CD34-EF56-GH78" --dry-run
```

## 输出

正常完成后，终端会报告发现频道数、清除频道数和虚拟时间，并在 `logs` 目录生成：

- `practice-p3-...jsonl`：完整请求、响应及策略行为日志；
- `practice-p3-...-summary.json`：本局结果摘要。

摘要中的 `overall_success=true` 表示策略完成且 `/exit` 已得到确认。

## 注意

官方本机接口不会告诉客户端当前页面属于演练还是正式测试，因此无门禁代码无法
从接口中识别测试类型。运行命令前，需要由操作者自行确认当前模拟器页面正确。
同一局只运行一个客户端，程序未结束时不要重复启动。
