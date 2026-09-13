from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from protocol import (
    ClientSettings,
    HttpStatusError,
    JsonlLogger,
    ProtocolError,
    RequestRejected,
    SimulatorClient,
    SimulatorError,
    TransportUncertainError,
    _LOCAL_ONLY_OPENER,
    _RejectRedirectHandler,
)
from policy import RunSummary
from run_practice import (
    resolve_robot_id,
    run_live_practice,
    write_summary,
)


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self, amount: int | None = None) -> bytes:
        body = json.dumps(self.payload).encode("utf-8")
        return body if amount is None else body[:amount]


class ProtocolTests(unittest.TestCase):
    def make_logger(self, directory: str) -> JsonlLogger:
        return JsonlLogger(Path(directory) / "test.jsonl", "offline-test")

    def test_default_opener_disables_proxies_and_redirects(self) -> None:
        proxy_handlers = [
            handler
            for handler in _LOCAL_ONLY_OPENER.handlers
            if handler.__class__.__name__ == "ProxyHandler"
        ]
        # build_opener 会把 ProxyHandler({}) 优化为完全不安装代理处理器。
        self.assertEqual(proxy_handlers, [])
        self.assertTrue(
            any(
                isinstance(handler, _RejectRedirectHandler)
                for handler in _LOCAL_ONLY_OPENER.handlers
            )
        )

    def test_exact_payloads_and_unique_request_ids(self) -> None:
        responses = iter(
            [
                FakeResponse(
                    {
                        "accepted": True,
                        "real_timestamp_ms": 1,
                        "virtual_time_s": 0,
                        "max_virtual_duration_s": 360000,
                        "max_real_duration_s": 1200,
                        "remaining_real_duration_s": 1200,
                    }
                ),
                FakeResponse(
                    {
                        "accepted": True,
                        "real_timestamp_ms": 2,
                        "virtual_time_s": 5,
                        "measure_result": "direction",
                        "svd_deg": 12.34,
                    }
                ),
                FakeResponse(
                    {
                        "accepted": True,
                        "real_timestamp_ms": 3,
                        "virtual_time_s": 8,
                        "clear_result": "no_target_in_range",
                    }
                ),
                FakeResponse(
                    {
                        "accepted": True,
                        "real_timestamp_ms": 4,
                        "virtual_time_s": 8,
                        "exit_reason": "user_exit",
                    }
                ),
            ]
        )
        captured = []

        def opener(request, timeout):
            captured.append((request, timeout))
            return next(responses)

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321),
                self.make_logger(directory),
                session_tag="unit",
                opener=opener,
            )
            client.enter()
            client.measure(3.0, 4.0, 2)
            client.clear(3.0, 4.0, 2)
            client.exit()

        bodies = [json.loads(request.data.decode("utf-8")) for request, _ in captured]
        self.assertEqual([request.full_url.rsplit("/", 1)[-1] for request, _ in captured], ["enter", "measure", "clear", "exit"])
        self.assertTrue(all(request.get_method() == "POST" for request, _ in captured))
        for request, _ in captured:
            headers = {key.lower(): value for key, value in request.header_items()}
            self.assertEqual(headers["content-type"], "application/json")
            self.assertNotIn("content-encoding", headers)
        self.assertEqual(bodies[1]["position"], {"x": 3.0, "y": 4.0})
        self.assertEqual(bodies[1]["channel"], 2)
        self.assertEqual(len({body["request_id"] for body in bodies}), 4)
        self.assertTrue(all(body["arena_id"] == "default" for body in bodies))

    def test_network_retry_reuses_identical_request(self) -> None:
        calls = []

        def opener(request, timeout):
            calls.append(request.data)
            if len(calls) == 1:
                raise URLError("offline transient")
            return FakeResponse(
                {
                    "accepted": True,
                    "real_timestamp_ms": 1,
                    "virtual_time_s": 0,
                    "max_virtual_duration_s": 360000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": 1200,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321, network_retries=1),
                self.make_logger(directory),
                session_tag="retry",
                opener=opener,
            )
            client.enter()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])

    def test_action_before_enter_is_rejected_without_network(self) -> None:
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            raise AssertionError("不应访问网络")

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321),
                self.make_logger(directory),
                session_tag="state",
                opener=opener,
            )
            with self.assertRaises(SimulatorError):
                client.measure(0.0, 0.0, 1)
            with self.assertRaises(SimulatorError):
                client.clear(0.0, 0.0, 1)
        self.assertEqual(calls, [])

    def test_keyboard_interrupt_during_request_locks_new_actions(self) -> None:
        def opener(request, timeout):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321),
                self.make_logger(directory),
                session_tag="interrupt",
                opener=opener,
            )
            with self.assertRaises(KeyboardInterrupt):
                client.enter()
            self.assertFalse(client.action_state_certain)

    def test_nonstandard_nan_response_is_rejected_and_locks_actions(self) -> None:
        def opener(request, timeout):
            return FakeResponse(
                {
                    "accepted": True,
                    "real_timestamp_ms": 1,
                    "virtual_time_s": float("nan"),
                    "max_virtual_duration_s": 360000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": 1200,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321),
                self.make_logger(directory),
                session_tag="nan",
                opener=opener,
            )
            with self.assertRaises(ProtocolError):
                client.enter()
            self.assertFalse(client.action_state_certain)

    def test_enter_remaining_time_must_be_in_range(self) -> None:
        def opener(request, timeout):
            return FakeResponse(
                {
                    "accepted": True,
                    "real_timestamp_ms": 1,
                    "virtual_time_s": 0,
                    "max_virtual_duration_s": 360000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": 1200.5,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321),
                self.make_logger(directory),
                session_tag="time",
                opener=opener,
            )
            with self.assertRaises(ProtocolError):
                client.enter()
            self.assertTrue(client.entered)

    def test_fractional_remaining_time_is_rejected_by_integer_contract(self) -> None:
        def opener(request, timeout):
            return FakeResponse(
                {
                    "accepted": True,
                    "real_timestamp_ms": 1,
                    "virtual_time_s": 0,
                    "max_virtual_duration_s": 360000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": 1199.5,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321),
                self.make_logger(directory),
                session_tag="fractional-time",
                opener=opener,
            )
            with self.assertRaises(ProtocolError):
                client.enter()
        self.assertTrue(client.entered)

    def test_interruption_while_logging_transport_error_locks_actions(self) -> None:
        class InterruptingLogger:
            def write(self, event, **fields):
                if event == "transport_error":
                    raise KeyboardInterrupt

        def opener(request, timeout):
            raise URLError("offline")

        client = SimulatorClient(
            ClientSettings("TESTTEAM", port=54321),
            InterruptingLogger(),
            session_tag="log-interrupt",
            opener=opener,
        )
        with self.assertRaises(KeyboardInterrupt):
            client.enter()
        self.assertFalse(client.action_state_certain)

    def test_interruption_while_logging_accepted_enter_marks_session_active(self) -> None:
        class InterruptingLogger:
            def write(self, event, **fields):
                if event == "response":
                    raise KeyboardInterrupt

        def opener(request, timeout):
            return FakeResponse(
                {
                    "accepted": True,
                    "real_timestamp_ms": 1,
                    "virtual_time_s": 0,
                    "max_virtual_duration_s": 360000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": 1200,
                }
            )

        client = SimulatorClient(
            ClientSettings("TESTTEAM", port=54321),
            InterruptingLogger(),
            session_tag="response-log-interrupt",
            opener=opener,
        )
        with self.assertRaises(KeyboardInterrupt):
            client.enter()
        self.assertTrue(client.action_state_certain)
        self.assertTrue(client.entered)

    def test_interrupt_between_parse_and_enter_state_remains_conservative(self) -> None:
        class NullLogger:
            def write(self, event, **fields):
                pass

        def opener(request, timeout):
            return FakeResponse(
                {
                    "accepted": True,
                    "real_timestamp_ms": 1,
                    "virtual_time_s": 0,
                    "max_virtual_duration_s": 360000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": 1200,
                }
            )

        client = SimulatorClient(
            ClientSettings("TESTTEAM", port=54321),
            NullLogger(),
            session_tag="atomic-state",
            opener=opener,
        )
        with patch("protocol.time.monotonic", side_effect=[1.0, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                client.enter()
        self.assertFalse(client.action_state_certain)
        self.assertFalse(client.entered)

    def test_contradictory_non_200_accepted_true_remains_uncertain(self) -> None:
        def opener(request, timeout):
            return FakeResponse(
                {"accepted": True, "real_timestamp_ms": 1, "virtual_time_s": 0},
                status=500,
            )

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321),
                self.make_logger(directory),
                session_tag="bad-status",
                opener=opener,
            )
            with self.assertRaises(HttpStatusError):
                client.enter()
        self.assertFalse(client.action_state_certain)

    def test_accepted_false_is_rejected_but_state_is_certain(self) -> None:
        def opener(request, timeout):
            return FakeResponse(
                {
                    "accepted": False,
                    "real_timestamp_ms": 1,
                    "virtual_time_s": 0,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321),
                self.make_logger(directory),
                session_tag="rejected",
                opener=opener,
            )
            with self.assertRaises(RequestRejected):
                client.enter()
            self.assertTrue(client.action_state_certain)

    def test_all_network_retries_fail_and_lock_new_actions(self) -> None:
        def opener(request, timeout):
            raise URLError("still offline")

        with tempfile.TemporaryDirectory() as directory:
            client = SimulatorClient(
                ClientSettings("TESTTEAM", port=54321, network_retries=1),
                self.make_logger(directory),
                session_tag="offline",
                opener=opener,
            )
            with self.assertRaises(TransportUncertainError):
                client.enter()
            self.assertFalse(client.action_state_certain)

    def test_placeholder_robot_id_is_requested_without_changing_config(self) -> None:
        config = {
            "robot_id": "YOUR_TEAM_ID",
            "port": 2026,
            "timeout_s": 5.0,
            "network_retries": 2,
        }
        with patch("builtins.input", return_value="TEAM-0001"), patch(
            "builtins.print"
        ):
            robot_id = resolve_robot_id(config)
        self.assertEqual(robot_id, "TEAM-0001")
        self.assertEqual(config["robot_id"], "YOUR_TEAM_ID")

    def test_configured_robot_id_does_not_prompt(self) -> None:
        config = {
            "robot_id": "TEAM-0001",
            "port": 2026,
            "timeout_s": 5.0,
            "network_retries": 2,
        }
        with patch("builtins.input") as mocked_input:
            robot_id = resolve_robot_id(config)
        self.assertEqual(robot_id, "TEAM-0001")
        mocked_input.assert_not_called()

    def test_blank_interactive_robot_id_is_rejected(self) -> None:
        config = {
            "robot_id": "YOUR_TEAM_ID",
            "port": 2026,
            "timeout_s": 5.0,
            "network_retries": 2,
        }
        with patch("builtins.input", return_value="   "), patch("builtins.print"):
            with self.assertRaises(ValueError):
                resolve_robot_id(config)

    def test_full_path_runs_without_gate(self) -> None:
        events: list[str] = []
        summary = RunSummary(
            strategy="offline-order-test",
            completed_normally=True,
            stop_reason="done",
            stations_visited=7,
            discovered_channels=(1,),
            cleared_channels=(1,),
            pending_channels=(),
            measure_count=1,
            clear_attempt_count=1,
            clear_success_count=1,
            no_signal_measure_count=0,
            shared_observation_count=0,
            search_certified=True,
            certified_absent_channels=tuple(range(2, 21)),
            final_virtual_time_s=10.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config = {
                "robot_id": "TESTTEAM",
                "port": 2026,
                "timeout_s": 5.0,
                "network_retries": 2,
                "log_directory": str(Path(directory) / "logs"),
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            class FakeClient:
                def __init__(self, settings, logger, session_tag):
                    events.append("client")
                    self.entered = False
                    self.closed = False
                    self.action_state_certain = True

                def enter(self):
                    events.append("enter")
                    self.entered = True
                    return {"remaining_real_duration_s": 1200}

                def exit(self):
                    events.append("exit")
                    self.closed = True
                    return {"accepted": True, "exit_reason": "user_exit"}

            class FakePolicy:
                name = "offline-order-test"

                def __init__(self, *args, **kwargs):
                    pass

                def run(self):
                    return summary

            with patch("run_practice.SimulatorClient", FakeClient), patch(
                "run_practice.SearchFirstPolicy", FakePolicy
            ), patch("builtins.print"):
                status = run_live_practice(
                    config_path,
                    config,
                    "AB12-CD34-EF56-GH78",
                )

            self.assertEqual(status, 0)
            self.assertEqual(events, ["client", "enter", "exit"])
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertNotIn("allow_live_practice", saved)

    def test_placeholder_case_code_is_rejected_before_connection(self) -> None:
        config = {
            "robot_id": "TESTTEAM",
            "port": 2026,
            "timeout_s": 5.0,
            "network_retries": 2,
            "log_directory": "logs",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(SystemExit):
                run_live_practice(path, config, "XXXX-XXXX-XXXX-XXXX")

    def test_summary_does_not_misreport_failed_exit_as_overall_success(self) -> None:
        summary = RunSummary(
            strategy="offline-test",
            completed_normally=True,
            stop_reason="policy done",
            stations_visited=7,
            discovered_channels=(1,),
            cleared_channels=(1,),
            pending_channels=(),
            measure_count=2,
            clear_attempt_count=1,
            clear_success_count=1,
            no_signal_measure_count=0,
            shared_observation_count=0,
            search_certified=True,
            certified_absent_channels=tuple(range(2, 21)),
            final_virtual_time_s=10.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            write_summary(
                path,
                summary,
                None,
                exit_action_accepted=False,
                manual_abort_required=True,
                run_error=RuntimeError("exit failed"),
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(saved["policy_completed"])
        self.assertFalse(saved["overall_success"])
        self.assertTrue(saved["manual_abort_required"])


if __name__ == "__main__":
    unittest.main()
