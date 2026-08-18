"""Structured ``rosclaw sim cmu-are`` command surface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rosclaw.daemon.client import DaemonClient, DaemonClientError
from rosclaw.integrations.cmu_are.contracts import (
    CMU_ARE_BODY_ID,
    CMU_ARE_EXPLORE_SCHEMA,
    CMU_ARE_NAV_SCHEMA,
    CMU_ARE_STOP_SCHEMA,
    CmuAreContractError,
    body_snapshot_hash,
    default_places_path,
    load_places,
    load_safety_contract,
    resolve_target,
)
from rosclaw.integrations.cmu_are.executor import (
    CMU_EXPLORE_CAPABILITY,
    CMU_NAVIGATE_CAPABILITY,
    CMU_STOP_CAPABILITY,
)
from rosclaw.kernel import (
    ActionEnvelope,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)
from rosclaw.sim.assets import verify_assets

CMU_ARE_REQUIRED_ASSETS = frozenset({"gazebo_world", "local_planner", "ariadne2_exploration"})
_CMU_ARE_ASSET_ENV = {
    "ariadne2-model-checkpoint": "CMU_ARE_ARIADNE2_MODEL_CHECKPOINT_SOURCE",
    "local-planner-correspondences": "CMU_ARE_LOCAL_PLANNER_CORRESPONDENCES_SOURCE",
    "local-planner-paths": "CMU_ARE_LOCAL_PLANNER_PATHS_SOURCE",
    "local-planner-start-paths": "CMU_ARE_LOCAL_PLANNER_START_PATHS_SOURCE",
    "local-planner-path-list": "CMU_ARE_LOCAL_PLANNER_PATH_LIST_SOURCE",
    "vehicle-simulator-meshes": "CMU_ARE_VEHICLE_SIMULATOR_MESHES_SOURCE",
}


def dispatch_sim_argv(argv: list[str]) -> int | None:
    """Dispatch only the new simulation namespace."""

    if not argv or argv[0] != "sim":
        return None
    parser = _build_parser()
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit as exc:
        # Keep the dispatcher callable from tests and other Python entrypoints
        # while preserving argparse's normal help/error output.
        return int(exc.code or 0)
    if getattr(args, "sim_json", False):
        args.json = True
    handler = getattr(args, "sim_handler", None)
    if not callable(handler):
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except (
        CmuAreContractError,
        DaemonClientError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        return _emit_error(args, "CMU_ARE_COMMAND_FAILED", str(exc), 2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw sim")
    parser.add_argument("--json", dest="sim_json", action="store_true", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="sim_command")
    cmu = commands.add_parser("cmu-are", help="ROS1 Noetic CMU ARE simulation")
    cmu.add_argument("--json", dest="sim_json", action="store_true", help=argparse.SUPPRESS)
    cmu_commands = cmu.add_subparsers(dest="cmu_command")

    check = cmu_commands.add_parser("check", help="Check contracts, assets, and rosbridge")
    _common_check_args(check)
    check.set_defaults(sim_handler=_cmd_check)

    launch = cmu_commands.add_parser("launch", help="Launch the fixed Docker Compose simulation")
    launch.add_argument("--compose-file", type=Path, default=None, help=argparse.SUPPRESS)
    launch.add_argument("--asset-root", type=Path, default=None)
    launch.add_argument("--rosbridge-url", default=None)
    launch.add_argument("--json", action="store_true")
    launch.set_defaults(sim_handler=_cmd_launch)

    navigate = cmu_commands.add_parser(
        "navigate", help="Navigate to a registered place or coordinate"
    )
    navigate.add_argument("--place")
    navigate.add_argument("--x", type=float)
    navigate.add_argument("--y", type=float)
    navigate.add_argument("--z", type=float, default=0.0)
    navigate.add_argument("--speed", type=float, default=None, dest="speed_mps")
    navigate.add_argument("--tolerance", type=float, default=None, dest="tolerance_m")
    navigate.add_argument("--timeout", type=float, default=120.0, dest="timeout_sec")
    navigate.add_argument("--places", type=Path, default=None)
    navigate.add_argument("--json", action="store_true")
    navigate.set_defaults(sim_handler=_cmd_navigate)

    explore = cmu_commands.add_parser("explore", help="Control ARiADNE2 exploration")
    explore.add_argument("command", choices=("start", "pause", "resume", "stop"))
    explore.add_argument("--speed", type=float, default=None, dest="speed_mps")
    explore.add_argument("--timeout", type=float, default=30.0, dest="timeout_sec")
    explore.add_argument("--json", action="store_true")
    explore.set_defaults(sim_handler=_cmd_explore)

    stop = cmu_commands.add_parser("stop", help="Stop simulated motion through rosclawd")
    stop.add_argument("--timeout", type=float, default=30.0, dest="timeout_sec")
    stop.add_argument("--down", action="store_true", help="Also bring down the Compose stack")
    stop.add_argument("--json", action="store_true")
    stop.set_defaults(sim_handler=_cmd_stop)
    return parser


def _common_check_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset-manifest", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--rosbridge-url", default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compose_file(args: argparse.Namespace) -> Path:
    candidate = getattr(args, "compose_file", None)
    return (
        (candidate or (_repo_root() / "docker/ros1/docker-compose.ros1-are.yml"))
        .expanduser()
        .resolve()
    )


def _emit(args: argparse.Namespace, payload: dict[str, Any], *, code: int = 0) -> int:
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if payload.get("ok") is False:
            error = payload.get("error", {})
            print(f"[ROSClaw] {error.get('code', 'ERROR')}: {error.get('message', '')}")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return code


def _emit_error(args: argparse.Namespace, error_code: str, message: str, code: int) -> int:
    return _emit(args, {"ok": False, "error": {"code": error_code, "message": message}}, code=code)


def _rosbridge_endpoint(value: str) -> tuple[str, int]:
    """Validate and normalize the local-only rosbridge endpoint."""

    parsed = urlparse(value)
    if parsed.scheme not in {"ws", "wss"} or parsed.hostname is None:
        raise CmuAreContractError("rosbridge URL must be ws:// or wss:// with a host")
    if parsed.hostname.casefold() not in {"127.0.0.1", "localhost", "::1"}:
        raise CmuAreContractError("CMU ARE rosbridge must stay on the local host")
    try:
        port = parsed.port or (443 if parsed.scheme == "wss" else 9090)
    except ValueError as exc:
        raise CmuAreContractError("rosbridge URL has an invalid port") from exc
    if not 1 <= port <= 65535:
        raise CmuAreContractError("rosbridge port must be in [1, 65535]")
    return value, port


def _set_asset_mount_environment(
    environment: dict[str, str],
    assets: dict[str, Any],
    *,
    asset_root: str | Path | None,
) -> None:
    """Pass verified host paths to Compose without copying large assets."""

    if asset_root is not None:
        environment["CMU_ARE_ASSET_ROOT"] = str(Path(asset_root).expanduser().resolve())
    for entry in assets.get("assets", []):
        if not isinstance(entry, dict) or entry.get("status") != "ok":
            continue
        env_name = _CMU_ARE_ASSET_ENV.get(str(entry.get("asset_id")))
        resolved = entry.get("resolved_path")
        if env_name and isinstance(resolved, str) and resolved:
            environment[env_name] = resolved


def _cmd_check(args: argparse.Namespace) -> int:
    root = _repo_root()
    manifest = args.asset_manifest or (root / "docs/assets/cmu-are-assets.yaml")
    asset_root = args.asset_root or os.environ.get("CMU_ARE_ASSET_ROOT")
    payload: dict[str, Any] = {
        "schema_version": "rosclaw.cmu_are.check.v1",
        "body_id": CMU_ARE_BODY_ID,
        "ok": True,
        "contract": {},
        "assets": {},
        "rosbridge": None,
    }
    try:
        safety = load_safety_contract()
        places = load_places()
        payload["contract"] = {
            "ok": True,
            "safety_card_hash": safety.digest,
            "place_count": len(places),
            "body_snapshot_hash": body_snapshot_hash(safety=safety, asset_manifest=manifest),
        }
    except Exception as exc:  # noqa: BLE001 - report a structured contract failure
        payload["contract"] = {"ok": False, "error": str(exc)}
        payload["ok"] = False

    try:
        payload["assets"] = verify_assets(
            project_root=root,
            manifest_path=manifest,
            required_for=set(CMU_ARE_REQUIRED_ASSETS),
            asset_root=asset_root,
        )
        if not payload["assets"].get("ok", False) and not args.contract_only:
            payload["ok"] = False
    except Exception as exc:  # noqa: BLE001
        payload["assets"] = {"ok": False, "error": str(exc)}
        payload["ok"] = False

    if not args.contract_only:
        from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport
        from rosclaw.integrations.cmu_are.adapter import CmuAreRosbridgeAdapter

        endpoint = args.rosbridge_url or os.environ.get(
            "ROSCLAW_CMU_ARE_ROSBRIDGE_URL", "ws://127.0.0.1:9090"
        )
        adapter = CmuAreRosbridgeAdapter(
            RosbridgeTransport(endpoint=RosbridgeEndpoint.from_url(endpoint), max_retries=0)
        )
        try:
            payload["rosbridge"] = adapter.check(timeout_sec=args.timeout)
            if not payload["rosbridge"].get("ok", False):
                payload["ok"] = False
        except Exception as exc:  # noqa: BLE001
            payload["rosbridge"] = {"ok": False, "error": str(exc), "endpoint": endpoint}
            payload["ok"] = False
    else:
        payload["rosbridge"] = {"skipped": True, "reason": "contract_only"}
    payload["launchable"] = bool(payload["assets"].get("ok", False))
    return _emit(args, payload, code=0 if payload["ok"] else 2)


def _cmd_launch(args: argparse.Namespace) -> int:
    root = _repo_root()
    asset_root = args.asset_root or os.environ.get("CMU_ARE_ASSET_ROOT")
    assets = verify_assets(
        project_root=root,
        required_for=set(CMU_ARE_REQUIRED_ASSETS),
        asset_root=asset_root,
    )
    compose = _compose_file(args)
    if not assets.get("ok"):
        return _emit(
            args,
            {
                "ok": False,
                "error": {
                    "code": "CMU_ARE_ASSETS_MISSING",
                    "message": "large CMU ARE assets are missing or have incorrect hashes",
                },
                "assets": assets,
                "compose_file": str(compose),
            },
            code=2,
        )
    if not compose.is_file():
        return _emit_error(args, "CMU_ARE_COMPOSE_MISSING", str(compose), 2)
    environment = os.environ.copy()
    _set_asset_mount_environment(environment, assets, asset_root=asset_root)
    endpoint = args.rosbridge_url or environment.get(
        "ROSCLAW_CMU_ARE_ROSBRIDGE_URL", "ws://127.0.0.1:9090"
    )
    endpoint, port = _rosbridge_endpoint(endpoint)
    environment["ROSCLAW_CMU_ARE_ROSBRIDGE_URL"] = endpoint
    environment["ROSBRIDGE_PORT"] = str(port)
    subprocess.run(
        ["docker", "compose", "-f", str(compose), "config", "--quiet"],
        check=True,
        env=environment,
    )
    completed = subprocess.run(
        ["docker", "compose", "-f", str(compose), "up", "-d", "--build", "rosclaw"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    payload = {
        "ok": completed.returncode == 0,
        "compose_file": str(compose),
        "rosbridge_url": endpoint,
        "asset_root": environment.get("CMU_ARE_ASSET_ROOT"),
        "command": ["docker", "compose", "-f", str(compose), "up", "-d", "--build", "rosclaw"],
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    return _emit(args, payload, code=0 if payload["ok"] else 2)


def _make_action(
    *,
    capability: str,
    arguments: dict[str, Any],
    timeout_sec: float,
) -> ActionEnvelope:
    body_hash = body_snapshot_hash()
    session_id = f"cmu-are-cli-{uuid.uuid4().hex[:16]}"
    trace_id = f"trace_cmu_are_{uuid.uuid4().hex[:20]}"
    return ActionEnvelope(
        actor_id="rosclaw-sim-cli",
        agent_framework="rosclaw-cli",
        session_id=session_id,
        body_id=CMU_ARE_BODY_ID,
        body_snapshot_hash=body_hash,
        capability_id=capability,
        arguments=arguments,
        execution_mode=ExecutionMode.SHADOW,
        parent_trace_id=trace_id,
        expected_effect=arguments.get("expected_effect"),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED,
            timeout_sec=timeout_sec,
            fail_closed=True,
        ),
    )


def _submit_action(args: argparse.Namespace, action: ActionEnvelope) -> int:
    client = DaemonClient(timeout_sec=min(10.0, float(getattr(args, "timeout_sec", 30.0))))
    try:
        payload = client.request_action(action)
        terminal = client.wait_for_action(
            action.action_id, timeout_sec=action.verification_policy.timeout_sec
        )
        payload = terminal
        receipt = payload.get("receipt")
        if not isinstance(receipt, dict):
            try:
                receipt = client.get_execution_receipt(action.action_id)
                payload["receipt"] = receipt
            except DaemonClientError:
                pass
        ok = isinstance(receipt, dict) and receipt.get("final_state") in {"COMPLETED", "DEGRADED"}
        payload = {"ok": ok, "action_id": action.action_id, **payload}
        return _emit(args, payload, code=0 if ok else 3)
    except DaemonClientError as exc:
        return _emit_error(args, exc.code, exc.message, 2)


def _cmd_navigate(args: argparse.Namespace) -> int:
    safety = load_safety_contract()
    places = load_places(args.places or default_places_path())
    target = resolve_target(
        place=args.place,
        x=args.x,
        y=args.y,
        z=args.z,
        frame_id="map",
        places=places,
        safety=safety,
    )
    speed = safety.max_speed_mps if args.speed_mps is None else safety.clamp_speed(args.speed_mps)
    tolerance = (
        safety.default_tolerance_m
        if args.tolerance_m is None
        else safety.validate_tolerance(args.tolerance_m)
    )
    timeout = safety.validate_timeout(args.timeout_sec)
    action = _make_action(
        capability=CMU_NAVIGATE_CAPABILITY,
        arguments={
            "schema_version": CMU_ARE_NAV_SCHEMA,
            "target": target,
            "speed_mps": speed,
            "tolerance_m": tolerance,
            "timeout_sec": timeout,
            "expected_effect": {
                "kind": "navigate_to_waypoint",
                "final_frame": "map",
                "stop_required": True,
            },
        },
        timeout_sec=timeout,
    )
    return _submit_action(args, action)


def _cmd_explore(args: argparse.Namespace) -> int:
    safety = load_safety_contract()
    speed = safety.max_speed_mps if args.speed_mps is None else safety.clamp_speed(args.speed_mps)
    timeout = safety.validate_timeout(args.timeout_sec)
    action = _make_action(
        capability=CMU_EXPLORE_CAPABILITY,
        arguments={
            "schema_version": CMU_ARE_EXPLORE_SCHEMA,
            "command": args.command,
            "speed_mps": speed,
            "timeout_sec": timeout,
        },
        timeout_sec=timeout,
    )
    return _submit_action(args, action)


def _cmd_stop(args: argparse.Namespace) -> int:
    timeout = load_safety_contract().validate_timeout(args.timeout_sec)
    if timeout > 60:
        raise CmuAreContractError("stop timeout_sec must be in (0, 60]")
    action = _make_action(
        capability=CMU_STOP_CAPABILITY,
        arguments={"schema_version": CMU_ARE_STOP_SCHEMA, "timeout_sec": timeout},
        timeout_sec=timeout,
    )
    result = _submit_action(args, action)
    if result == 0 and args.down:
        compose = _compose_file(args)
        subprocess.run(["docker", "compose", "-f", str(compose), "down"], check=False)
    return result


__all__ = ["dispatch_sim_argv"]
