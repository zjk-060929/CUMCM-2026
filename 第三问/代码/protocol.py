"""CUMCM 2026 B 题模拟器的串行 HTTP 客户端。

本模块只访问本机回环地址 127.0.0.1。它不启动模拟器、不控制界面，
也无法从接口响应判断当前页面是演练测试还是正式测试。
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
import unicodedata
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


ALLOWED_PATHS = frozenset({"/enter", "/measure", "/clear", "/exit"})
MAX_JSON_BYTES = 65536


class _RejectRedirectHandler(HTTPRedirectHandler):
    """拒绝 30x，防止本机服务把请求重定向到非回环地址。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_LOCAL_ONLY_OPENER = build_opener(ProxyHandler({}), _RejectRedirectHandler())


def _local_only_urlopen(request: Request, timeout: float):
    """禁用环境代理并禁用重定向，只打开请求中写死的回环 URL。"""

    return _LOCAL_ONLY_OPENER.open(request, timeout=timeout)


class SimulatorError(RuntimeError):
    """模拟器通信或协议错误。"""


class TransportUncertainError(SimulatorError):
    """重试后仍未拿到响应，此时最后一个动作的执行状态未知。"""


class HttpStatusError(SimulatorError):
    """模拟器返回非 200 状态。"""

    def __init__(self, status: int, response: Any):
        super().__init__(f"HTTP {status}: {response!r}")
        self.status = status
        self.response = response


class RequestRejected(SimulatorError):
    """HTTP 200 但 accepted=false。"""

    def __init__(self, response: dict[str, Any]):
        super().__init__(f"请求未被模拟器接受: {response}")
        self.response = response


class ProtocolError(SimulatorError):
    """响应不是附件 2 规定的 JSON 结构。"""


def _json_safe(value: Any) -> Any:
    """把日志字段转换为 JSON 可序列化形式。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return value


class JsonlLogger:
    """逐行落盘，避免程序异常时丢失整场记录。"""

    def __init__(self, path: Path, case_code: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.case_code = case_code

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "monotonic_s": time.monotonic(),
            "case_code": self.case_code,
            "event": event,
            **{key: _json_safe(value) for key, value in fields.items()},
        }
        line = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()


def _validate_identifier(name: str, value: str, byte_limit: int) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} 必须是非空字符串")
    if len(value.encode("utf-8")) > byte_limit:
        raise ValueError(f"{name} 的 UTF-8 长度不得超过 {byte_limit} 字节")
    for character in value:
        if unicodedata.category(character) in {"Cc", "Cf"}:
            raise ValueError(f"{name} 不能包含控制字符或不可见格式字符")


def _strict_response(raw: bytes) -> dict[str, Any]:
    def reject_nonstandard_number(token: str) -> None:
        raise ValueError(f"JSON 不允许常量 {token}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 含重复键 {key}")
            result[key] = value
        return result

    try:
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError(f"响应体超过 {MAX_JSON_BYTES} 字节")
        decoded = raw.decode("utf-8")
        response = json.loads(
            decoded,
            parse_constant=reject_nonstandard_number,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ProtocolError("响应不是合法的 UTF-8 JSON") from error
    if not isinstance(response, dict):
        raise ProtocolError("响应 JSON 顶层必须是对象")
    for key in ("accepted", "real_timestamp_ms", "virtual_time_s"):
        if key not in response:
            raise ProtocolError(f"响应缺少字段 {key}")
    if not isinstance(response["accepted"], bool):
        raise ProtocolError("accepted 必须是 boolean")
    for key in ("real_timestamp_ms", "virtual_time_s"):
        value = response[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(f"{key} 必须是 number")
        if not math.isfinite(float(value)):
            raise ProtocolError(f"{key} 必须是有限数值")
        if float(value) < 0:
            raise ProtocolError(f"{key} 不能是负数")
    if response["accepted"] is False and set(response) != {
        "accepted",
        "real_timestamp_ms",
        "virtual_time_s",
    }:
        raise ProtocolError("accepted=false 响应只能包含三个公共字段")
    return response


@dataclass(frozen=True)
class ClientSettings:
    robot_id: str
    port: int = 2026
    timeout_s: float = 5.0
    network_retries: int = 2

    def validate(self) -> None:
        _validate_identifier("robot_id", self.robot_id, 64)
        if self.robot_id == "YOUR_TEAM_ID" or "参赛队号" in self.robot_id:
            raise ValueError("请先把 robot_id 改成模拟器当前登录的参赛队号")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("port 必须是 1 至 65535 的整数")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)):
            raise ValueError("timeout_s 必须是数值")
        if not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("timeout_s 必须为正的有限数值")
        if float(self.timeout_s) > 5.0:
            raise ValueError("演练安全配置要求 timeout_s 不得超过 5 秒")
        if (
            isinstance(self.network_retries, bool)
            or not isinstance(self.network_retries, int)
            or self.network_retries < 0
            or self.network_retries > 2
        ):
            raise ValueError("network_retries 必须是 0 至 2 的整数")


class SimulatorClient:
    """严格串行、带幂等重试的本地客户端。"""

    def __init__(
        self,
        settings: ClientSettings,
        logger: JsonlLogger,
        session_tag: str,
        opener: Callable[..., Any] = _local_only_urlopen,
    ):
        settings.validate()
        _validate_identifier("session_tag", session_tag, 40)
        self.settings = settings
        self.logger = logger
        self.session_tag = session_tag
        self._opener = opener
        self._counter = 0
        self.entered = False
        self.closed = False
        self.action_state_certain = True
        self.last_virtual_time_s = 0.0
        self.last_action_started_monotonic_s: float | None = None
        self.last_response_monotonic_s: float | None = None
        self._request_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        # 故意不允许配置主机名，避免把竞赛接口暴露到网络或误连远端。
        return f"http://127.0.0.1:{int(self.settings.port)}"

    @property
    def worst_case_request_wait_s(self) -> float:
        return self.settings.timeout_s * (self.settings.network_retries + 1)

    def _next_request_id(self, action: str) -> str:
        self._counter += 1
        request_id = f"p3-practice-{self.session_tag}-{self._counter:04d}-{action}"
        _validate_identifier("request_id", request_id, 128)
        return request_id

    def _base_payload(self, action: str) -> dict[str, Any]:
        return {
            "arena_id": "default",
            "robot_id": self.settings.robot_id,
            "request_id": self._next_request_id(action),
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._request_lock.acquire(blocking=False):
            raise SimulatorError("禁止并发发送不同动作；请等待上一HTTP响应完成")
        try:
            return self._post_serial(path, payload)
        finally:
            self._request_lock.release()

    def _post_serial(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path not in ALLOWED_PATHS:
            raise ValueError(f"不允许的接口路径: {path}")
        if not self.action_state_certain:
            raise TransportUncertainError(
                "上一动作状态未知，禁止发送不同的新动作；请在演练界面手工中止本局"
            )

        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > MAX_JSON_BYTES:
            raise ValueError(f"请求体超过附件 2 规定的 {MAX_JSON_BYTES} 字节")

        self.logger.write("request", path=path, payload=payload)
        last_error: BaseException | None = None
        action_started_monotonic_s = time.monotonic()
        for attempt in range(self.settings.network_retries + 1):
            request = Request(
                self.base_url + path,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # 从真正准备发送开始即视为“状态未知”。只有拿到完整且可解析的
            # HTTP 响应后才能恢复；这样即使在异常处理或日志写入时被 Ctrl+C
            # 打断，也绝不会误发一个不同的新动作。
            self.action_state_certain = False
            try:
                try:
                    with self._opener(
                        request, timeout=self.settings.timeout_s
                    ) as http_response:
                        status = int(http_response.getcode())
                        raw = http_response.read(MAX_JSON_BYTES + 1)
                except HTTPError as http_error:
                    status = int(http_error.code)
                    raw = http_error.read(MAX_JSON_BYTES + 1)
                response = _strict_response(raw)
            except (
                URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                HTTPException,
            ) as error:
                last_error = error
                self.logger.write(
                    "transport_error",
                    path=path,
                    attempt=attempt + 1,
                    will_retry=attempt < self.settings.network_retries,
                    error=error,
                )
                continue
            except BaseException as error:
                # 请求可能已经到达模拟器，但没有取得可判定的完整响应。
                # 此后不得发送不同的新动作。
                self.action_state_certain = False
                try:
                    self.logger.write(
                        "uncertain_response_error",
                        path=path,
                        attempt=attempt + 1,
                        error=error,
                    )
                except Exception:
                    pass
                raise

            if status != 200:
                # 明确 rejected 的完整响应不会创建动作；若出现非200却
                # accepted=true 的矛盾响应，则继续保持 unknown，保守处置。
                if response["accepted"] is False:
                    self.action_state_certain = True
                self.logger.write(
                    "response",
                    path=path,
                    attempt=attempt + 1,
                    status=status,
                    response=response,
                )
                raise HttpStatusError(status, response)
            if response["accepted"] is not True:
                self.action_state_certain = True
                self.logger.write(
                    "response",
                    path=path,
                    attempt=attempt + 1,
                    status=status,
                    response=response,
                )
                raise RequestRejected(response)
            virtual_time_s = float(response["virtual_time_s"])
            if virtual_time_s + 1.0e-9 < self.last_virtual_time_s:
                raise ProtocolError("accepted=true 的虚拟时钟发生倒退")
            self.last_virtual_time_s = virtual_time_s
            self.last_action_started_monotonic_s = action_started_monotonic_s
            self.last_response_monotonic_s = time.monotonic()
            # 会话状态在收到 accepted=true 后立即更新，不能让随后可能发生的
            # 本地日志/输出异常掩盖服务器上已存在或已结束的会话。
            if path == "/enter":
                self.entered = True
            elif path == "/exit":
                self.closed = True
            # 必须最后才恢复 certain：若 Ctrl+C 落在解析响应和更新会话状态
            # 之间，调用方只会收到保守的人工中止提示，不会漏掉活跃会话。
            self.action_state_certain = True
            self.logger.write(
                "response",
                path=path,
                attempt=attempt + 1,
                status=status,
                response=response,
            )
            return response

        self.action_state_certain = False
        raise TransportUncertainError(
            "同一 request_id 的幂等重试全部失败，动作是否执行未知；"
            "禁止继续自动发新动作，请在演练界面手工中止"
        ) from last_error

    def enter(self) -> dict[str, Any]:
        if self.entered or self.closed:
            raise SimulatorError("当前客户端状态不允许再次 /enter")
        response = self._post("/enter", self._base_payload("enter"))
        # _post 已在收到 accepted=true 时标记 entered；即使后续字段校验异常，
        # 调用方也能安全决定是否发送 /exit。
        for key in (
            "max_virtual_duration_s",
            "max_real_duration_s",
            "remaining_real_duration_s",
        ):
            if key not in response or isinstance(response[key], bool) or not isinstance(
                response[key], (int, float)
            ):
                raise ProtocolError(f"/enter 响应缺少合法字段 {key}")
            if not math.isfinite(float(response[key])):
                raise ProtocolError(f"/enter 响应字段 {key} 必须是有限数值")
        if float(response["max_virtual_duration_s"]) <= 0:
            raise ProtocolError("max_virtual_duration_s 必须为正数")
        if float(response["max_real_duration_s"]) <= 0:
            raise ProtocolError("max_real_duration_s 必须为正数")
        remaining = float(response["remaining_real_duration_s"])
        if not remaining.is_integer() or not 0.0 <= remaining <= 1200.0:
            raise ProtocolError(
                "remaining_real_duration_s 必须是 0 至 1200 的整数"
            )
        return response

    def measure(self, x: float, y: float, channel: int) -> dict[str, Any]:
        if not self.entered or self.closed:
            raise SimulatorError("只有成功 /enter 后且 /exit 前才能调用 /measure")
        self._validate_action(x, y, channel)
        payload = self._base_payload("measure")
        payload.update(
            {"position": {"x": float(x), "y": float(y)}, "channel": int(channel)}
        )
        response = self._post("/measure", payload)
        result = response.get("measure_result")
        if result not in {"no_signal", "near", "direction"}:
            raise ProtocolError(f"未知 measure_result: {result!r}")
        if result == "direction":
            bearing = response.get("svd_deg")
            if isinstance(bearing, bool) or not isinstance(bearing, (int, float)):
                raise ProtocolError("direction 响应缺少数值 svd_deg")
            if not 0.0 <= float(bearing) < 360.0:
                raise ProtocolError("svd_deg 不在 [0, 360) 内")
        elif "svd_deg" in response:
            raise ProtocolError("非 direction 响应不应包含 svd_deg")
        return response

    def clear(self, x: float, y: float, channel: int) -> dict[str, Any]:
        if not self.entered or self.closed:
            raise SimulatorError("只有成功 /enter 后且 /exit 前才能调用 /clear")
        self._validate_action(x, y, channel)
        payload = self._base_payload("clear")
        payload.update(
            {"position": {"x": float(x), "y": float(y)}, "channel": int(channel)}
        )
        response = self._post("/clear", payload)
        if response.get("clear_result") not in {"success", "no_target_in_range"}:
            raise ProtocolError(f"未知 clear_result: {response.get('clear_result')!r}")
        return response

    def exit(self) -> dict[str, Any]:
        if not self.entered or self.closed:
            raise SimulatorError("当前客户端状态不允许 /exit")
        response = self._post("/exit", self._base_payload("exit"))
        # _post 已在收到 accepted=true 时标记 closed；附加字段异常也不能再次 /exit。
        if response.get("exit_reason") != "user_exit":
            raise ProtocolError(f"未知 exit_reason: {response.get('exit_reason')!r}")
        return response

    @staticmethod
    def _validate_action(x: float, y: float, channel: int) -> None:
        for name, value in (("x", x), ("y", y)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数值")
            if not math.isfinite(float(value)) or abs(float(value)) > 2_000_000:
                raise ValueError(f"{name} 必须是绝对值不超过 2000000 的有限数值")
        if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20:
            raise ValueError("channel 必须是 1 至 20 的整数")
