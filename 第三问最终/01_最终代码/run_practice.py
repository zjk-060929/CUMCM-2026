"""问题3 Jammer 客户端无门禁入口。

提供测试案例编码后，程序直接连接本机模拟器并执行策略。
程序不启动或操作模拟器界面，也不访问远程地址。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from geometry import (
    COVERAGE_LAYOUTS,
    coverage_route_length,
    coverage_stations,
    minimax_coverage_radius,
    worst_coverage_distance,
)
from policy import RunSummary, SearchFirstPolicy
from adaptive_policy import CorrelatedAdaptivePolicy
from joint_policy import JointScale125Policy
from protocol import (
    ClientSettings,
    JsonlLogger,
    SimulatorClient,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
ALLOWED_CONFIG_KEYS = {
    "robot_id",
    "port",
    "timeout_s",
    "network_retries",
    "coverage_layout",
    "shared_observations",
    "origin_information_scan",
    "opportunistic_during_search",
    "strategy",
    "log_directory",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("config.json 顶层必须是 JSON 对象")
    unknown = set(config) - ALLOWED_CONFIG_KEYS
    if unknown:
        raise ValueError(f"config.json 含未知字段: {sorted(unknown)}")
    required = {"robot_id", "port", "timeout_s", "network_retries"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"config.json 缺少字段: {sorted(missing)}")
    if not isinstance(config.get("opportunistic_during_search", True), bool):
        raise ValueError("opportunistic_during_search 必须是 true 或 false")
    if config.get("coverage_layout", "heptagon_minimax") not in COVERAGE_LAYOUTS:
        raise ValueError(f"coverage_layout 必须是以下之一: {COVERAGE_LAYOUTS}")
    if not isinstance(config.get("shared_observations", True), bool):
        raise ValueError("shared_observations 必须是 true 或 false")
    if not isinstance(config.get("origin_information_scan", True), bool):
        raise ValueError("origin_information_scan 必须是 true 或 false")
    if config.get("strategy", "baseline") not in {
        "baseline",
        "joint_scale125",
        "correlated_adaptive4",
    }:
        raise ValueError(
            "strategy 必须是 baseline、joint_scale125 或 correlated_adaptive4"
        )
    return config


def safe_case_tag(case_code: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", case_code).strip("-")
    return (tag or "case")[:32]


def print_dry_run(config_path: Path, config: dict[str, Any]) -> None:
    layout = config.get("coverage_layout", "heptagon_minimax")
    print("离线检查模式：不会建立网络连接，也不会调用 /enter。")
    print(f"配置文件：{config_path}")
    print(f"当前端口：{config['port']}")
    print("实时演练配置：无需逐局修改 config.json")
    print(
        "搜索中机会式清除："
        f"{config.get('opportunistic_during_search', True)}"
    )
    print(f"策略：{config.get('strategy', 'baseline')}")
    print(f"共用停靠点补测：{config.get('shared_observations', True)}")
    print(f"原点零移动预扫描：{config.get('origin_information_scan', True)}")
    print(f"七点布局：{layout}")
    print(f"正七边形解析半径：{minimax_coverage_radius():.6f} m")
    print(f"最坏最近距离：{worst_coverage_distance(layout):.6f} m")
    print(f"保证性搜索骨架长度：{coverage_route_length(layout):.6f} m")
    print("七个检测点：")
    for index, point in enumerate(coverage_stations(layout), 1):
        print(f"  H{index}: ({point[0]:.6f}, {point[1]:.6f})")
    if config["robot_id"] == "YOUR_TEAM_ID":
        print(
            "尚未填写 robot_id；这是安全的。实时演练时程序会在联网前"
            "要求你根据模拟器右上角现场输入参赛队号，不需要把队号写进配置文件。"
        )


def resolve_robot_id(config: dict[str, Any]) -> str:
    """取得本局参赛队号；占位值只允许在人工交互终端中现场输入。"""

    configured = config["robot_id"]
    if configured == "YOUR_TEAM_ID" or (
        isinstance(configured, str) and "参赛队号" in configured
    ):
        print()
        print("请查看模拟器右上角，输入当前登录的参赛队号（不是登录密码）：")
        robot_id = input("> ").strip()
    else:
        robot_id = configured

    settings = ClientSettings(
        robot_id=robot_id,
        port=config["port"],
        timeout_s=config["timeout_s"],
        network_retries=config["network_retries"],
    )
    settings.validate()
    return robot_id


def write_summary(
    path: Path,
    summary: RunSummary,
    exit_response: Any,
    *,
    exit_action_accepted: bool,
    manual_abort_required: bool,
    run_error: BaseException | None,
) -> None:
    data = summary.as_dict()
    data["exit_response"] = exit_response
    data["policy_completed"] = summary.completed_normally
    data["exit_action_accepted"] = exit_action_accepted
    data["exit_response_verified"] = exit_response is not None
    data["manual_abort_required"] = manual_abort_required
    data["overall_success"] = bool(
        summary.completed_normally
        and exit_action_accepted
        and exit_response is not None
        and not manual_abort_required
        and run_error is None
    )
    data["run_error"] = (
        None
        if run_error is None
        else {"type": type(run_error).__name__, "message": str(run_error)}
    )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def best_effort_error_log(logger: JsonlLogger, event: str, **fields: Any) -> None:
    """错误处置不能再被磁盘故障或二次 Ctrl+C 打断。"""

    try:
        logger.write(event, **fields)
    except BaseException as log_error:
        try:
            print(f"本地日志写入失败：{type(log_error).__name__}: {log_error}")
        except BaseException:
            pass


def run_live_practice(config_path: Path, config: dict[str, Any], case_code: str) -> int:
    case_code = case_code.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}", case_code):
        raise SystemExit(
            "--case-code 必须填写数据准备完成后界面显示的四组实际测试案例编码，"
            "例如 AB12-CD34-EF56-GH78"
        )
    if case_code == "XXXX-XXXX-XXXX-XXXX":
        raise SystemExit("不能使用数据准备阶段的 XXXX 占位符；请等待实际案例编码")
    if any(ord(character) < 32 or ord(character) == 127 for character in case_code):
        raise SystemExit("测试案例编码不能包含控制字符")

    robot_id = resolve_robot_id(config)
    policy_class = {
        "baseline": SearchFirstPolicy,
        "joint_scale125": JointScale125Policy,
        "correlated_adaptive4": CorrelatedAdaptivePolicy,
    }[config.get("strategy", "baseline")]
    settings = ClientSettings(
        robot_id=robot_id,
        port=config["port"],
        timeout_s=config["timeout_s"],
        network_retries=config["network_retries"],
    )
    settings.validate()
    tag = safe_case_tag(case_code)
    session_tag = f"{tag[:18]}-{uuid.uuid4().hex[:8]}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_directory = Path(config.get("log_directory", "logs"))
    if not log_directory.is_absolute():
        log_directory = config_path.resolve().parent / log_directory
    log_path = log_directory / f"practice-p3-{tag}-{timestamp}.jsonl"
    summary_path = log_directory / f"practice-p3-{tag}-{timestamp}-summary.json"
    logger = JsonlLogger(log_path, case_code=case_code)
    logger.write(
        "practice_run_started",
        config_path=config_path.resolve(),
        base_url=f"http://127.0.0.1:{settings.port}",
        strategy=policy_class.name,
    )

    client = SimulatorClient(settings, logger, session_tag=session_tag)
    policy: SearchFirstPolicy | None = None
    summary: RunSummary | None = None
    exit_response: Any = None
    run_error: BaseException | None = None

    try:
        enter_response = client.enter()
        print(
            "已进入问题3演练接口；本局实际剩余现实时间："
            f"{enter_response['remaining_real_duration_s']} 秒"
        )
        policy = policy_class(
            client,
            logger,
            enter_response,
            opportunistic_during_search=config.get(
                "opportunistic_during_search", True
            ),
            coverage_layout=config.get("coverage_layout", "heptagon_minimax"),
            shared_observations=config.get("shared_observations", True),
            origin_information_scan=config.get("origin_information_scan", True),
        )
        summary = policy.run()
        print(
            f"策略结束：发现 {len(summary.discovered_channels)} 个频道，"
            f"清除 {len(summary.cleared_channels)} 个频道，"
            f"虚拟时间 {summary.final_virtual_time_s:.6f} 秒。"
        )
    except KeyboardInterrupt as error:
        run_error = error
        best_effort_error_log(logger, "run_interrupted", reason="KeyboardInterrupt")
        print("收到键盘中断。")
    except BaseException as error:
        run_error = error
        best_effort_error_log(logger, "run_failed", error=error)
        print(f"演练程序停止：{type(error).__name__}: {error}")
    finally:
        if client.entered and not client.closed:
            if client.action_state_certain:
                try:
                    exit_response = client.exit()
                    print("已正常发送 /exit。")
                except BaseException as exit_error:
                    best_effort_error_log(logger, "exit_failed", error=exit_error)
                    print(f"/exit 失败：{type(exit_error).__name__}: {exit_error}")
                    if run_error is None:
                        run_error = exit_error
            else:
                # 统一在 finally 之后给出一次人工处置提示。
                pass

    manual_abort_required = (not client.action_state_certain) or (
        client.entered and not client.closed
    )
    if manual_abort_required:
        print(
            "重要：/enter 或最后一个动作可能已经执行，程序按照协议没有发送"
            "不同的新动作。请先确认客户端已停止，再核对当前会话标题仍为"
            "“问题3演练测试”，然后在该演练页面点击红色“中止测试”并完成"
            "两次确认。"
        )

    if summary is None and policy is not None:
        summary = policy.interrupted_summary(
            f"{type(run_error).__name__}: {run_error}" if run_error else "未知中断"
        )
    if summary is not None:
        write_summary(
            summary_path,
            summary,
            exit_response,
            exit_action_accepted=client.closed,
            manual_abort_required=manual_abort_required,
            run_error=run_error,
        )
        print(f"结果摘要：{summary_path}")
    print(f"明文日志：{log_path}")

    if manual_abort_required:
        return 3
    return 0 if run_error is None else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUMCM 2026 B题问题3演练客户端")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--case-code",
        required=True,
        help="模拟器问题3演练页面显示的测试案例编码，仅写入本地日志",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印配置和七点布局，不连接模拟器",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config_path = arguments.config.resolve()
    config = load_config(config_path)
    if arguments.dry_run:
        print_dry_run(config_path, config)
        return 0
    return run_live_practice(config_path, config, arguments.case_code)


if __name__ == "__main__":
    sys.exit(main())
