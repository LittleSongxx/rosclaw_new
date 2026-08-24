"""CLI handlers for the ROSClaw Skill Hub lifecycle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.skill.builtins import get_builtin_skill, list_builtin_skills
from rosclaw.skill.catalog import submit_to_catalog
from rosclaw.skill.eval import evaluate_skill
from rosclaw.skill.mining import mine_skill_candidate
from rosclaw.skill.models import SkillPackage, SkillRef
from rosclaw.skill.package import (
    package_skill,
    prepare_manifest,
    scan_forbidden_content,
    verify_package,
)
from rosclaw.skill.promote import promote_candidate
from rosclaw.skill.registry import SkillLocalRegistry
from rosclaw.skill.rollback import rollback_skill
from rosclaw.skill.upload import upload_skill
from rosclaw.skill.validators import validate_package

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def _resolve_skill_dir(name: str, workspace: str | None = None, cwd_fallback: bool = True) -> Path:
    if workspace:
        return Path(workspace).expanduser().resolve() / name
    # Prefer ~/.rosclaw/skills/NAME, otherwise current dir.
    home_skills = Path(resolve_home(None)) / "skills" / name
    if cwd_fallback and (Path.cwd() / name).exists():
        return Path.cwd() / name
    return home_skills


def _copy_template(template_dir: Path, dest: Path, context: dict[str, str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src_path in sorted(template_dir.rglob("*")):
        if not src_path.is_file():
            continue
        if "__pycache__" in src_path.parts or src_path.suffix in {".pyc", ".pyo"}:
            continue
        rel = src_path.relative_to(template_dir)
        dst_path = dest / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        text = src_path.read_text(encoding="utf-8")
        # Simple brace substitution (str.format would be too strict with JSON braces).
        for key, value in context.items():
            text = text.replace(f"{{{key}}}", value)
        dst_path.write_text(text, encoding="utf-8")
    # Create placeholder files for empty dirs.
    for placeholder in [
        "policies/checkpoints/.gitkeep",
        "evidence/practice/.gitkeep",
        "evidence/eval/.gitkeep",
        "evidence/videos/.gitkeep",
        "evidence/reports/.gitkeep",
        "evidence/signatures/.gitkeep",
        ".rosclaw/package/.gitkeep",
    ]:
        ph = dest / placeholder
        if not ph.exists():
            ph.parent.mkdir(parents=True, exist_ok=True)
            ph.write_text("", encoding="utf-8")


def _init_context(name: str, robot: str, category: str, namespace: str) -> dict[str, str]:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return {
        "name": name,
        "display_name": name.replace("_", " ").replace("-", " ").title(),
        "robot": robot,
        "category": category,
        "namespace": namespace,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "created_at_date": now.strftime("%Y-%m-%d"),
        "description": f"A reusable ROSClaw skill for {name}.",
    }


def cmd_skill_init(args: argparse.Namespace) -> int:
    name = args.name
    robot = args.robot or "unitree_g1"
    category = args.category or "manipulation"
    namespace = args.namespace or "ros-claw"
    output = Path(args.output).expanduser().resolve() if args.output else _resolve_skill_dir(name)
    template = args.template or "default"

    template_dir = Path(__file__).parent / "templates" / template
    if not template_dir.exists():
        print(f"[ROSClaw] Template not found: {template}")
        return 1

    if output.exists() and not args.force:
        print(f"[ROSClaw] Skill directory already exists: {output}")
        print("[ROSClaw] Use --force to overwrite")
        return 1

    if output.exists():
        shutil.rmtree(output)

    context = _init_context(name, robot, category, namespace)
    _copy_template(template_dir, output, context)

    # Run initial validation.
    pkg = SkillPackage(output).try_load()
    report = validate_package(pkg)

    # Register.
    registry = SkillLocalRegistry()
    registry.add(pkg)

    print(f"[ROSClaw] Created skill package: {output}")
    if report.warnings:
        for w in report.warnings:
            print(f"[ROSClaw] Warning: {w}")
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def _load_skill_dir_arg(args: argparse.Namespace) -> Path:
    if getattr(args, "skill_dir", None):
        path = Path(args.skill_dir).expanduser()
        if path.exists() or path.is_absolute() or "/" in args.skill_dir or "\\" in args.skill_dir:
            return path.resolve()
        # Treat as skill name.
        return _resolve_skill_dir(args.skill_dir, workspace=getattr(args, "workspace", None))
    if getattr(args, "name", None):
        return _resolve_skill_dir(args.name, workspace=getattr(args, "workspace", None))
    raise ValueError("No skill directory or name provided")


def cmd_skill_validate(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()
    report = validate_package(pkg)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "checks": report.checks,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"[ROSClaw] Validating {skill_dir.name}")
        for check, ok in report.checks.items():
            print(f"  {'✓' if ok else '✗'} {check}")
        for e in report.errors:
            print(f"  ✗ {e}")
        for w in report.warnings:
            print(f"  ! {w}")
        print(f"[ROSClaw] Result: {'PASS' if report.ok else 'FAIL'}")
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# Mine
# ---------------------------------------------------------------------------


def _resolve_output_or_name(value: str | None) -> Path:
    if not value:
        return Path(resolve_home(None)) / "skills"
    path = Path(value).expanduser()
    if path.exists() or path.is_absolute() or "/" in value or "\\" in value:
        return path.resolve()
    return _resolve_skill_dir(value)


def cmd_skill_mine(args: argparse.Namespace) -> int:
    source_dir = Path(args.source).expanduser().resolve()
    output = _resolve_output_or_name(args.output) if args.output else _resolve_skill_dir(args.task)
    if not output.exists():
        print(f"[ROSClaw] Skill output directory does not exist: {output}")
        return 1
    pkg = SkillPackage(output).try_load()
    report = mine_skill_candidate(pkg, source_dir, candidate_id=args.candidate)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Mined candidate {report.candidate_id}")
        print(f"  source episodes: {len(report.source_episodes)}")
        print(f"  score: {report.score}")
        print(f"  generated: {', '.join(report.generated_files)}")
    return 0


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def cmd_skill_eval(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()
    report = evaluate_skill(
        pkg, candidate_id=args.candidate, mode=args.mode, save_evidence=args.save_evidence
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Eval {report.skill}@{report.candidate_id or report.version}")
        for check, ok in report.checks.items():
            print(f"  {'✓' if ok else '✗'} {check}")
        print(f"  metrics: {json.dumps(report.metrics, ensure_ascii=False)}")
        print(f"[ROSClaw] Decision: {report.decision.upper()}")
        if report.artifacts:
            print(f"  report: {report.artifacts.get('report')}")
    return 0 if report.decision == "pass" else 1


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------


def cmd_skill_promote(args: argparse.Namespace) -> int:
    ref = SkillRef(args.skill_ref)
    skill_dir = _resolve_skill_dir(ref.name, workspace=getattr(args, "workspace", None))
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()
    candidate_id = ref.candidate_id or (pkg.skill.metadata.candidate_id if pkg.skill else None)
    if not candidate_id:
        print("[ROSClaw] No candidate specified")
        return 1
    try:
        result = promote_candidate(
            pkg,
            candidate_id,
            to_version=args.to_version,
            stage=args.stage,
            require_eval_pass=args.require_eval_pass,
        )
    except ValueError as exc:
        print(f"[ROSClaw] Promotion blocked: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(
            f"[ROSClaw] Promoted {ref.name}@{candidate_id} to v{result['version']} ({result['stage']})"
        )
        print(f"  package_hash: {result['package_hash']}")
    return 0


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------


def cmd_skill_package(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()

    # Pre-package forbidden content scan.
    secrets, paths = scan_forbidden_content(skill_dir)
    if secrets:
        print("[ROSClaw] Secret scan failed:")
        for s in secrets:
            print(f"  {s}")
        return 1
    if paths:
        print("[ROSClaw] Absolute path warnings:")
        for p in paths:
            print(f"  {p}")

    archive = package_skill(
        pkg,
        output_dir=Path(args.output),
        include_evidence=args.include_evidence,
        format=args.format,
    )

    if args.json:
        print(
            json.dumps(
                {"archive": str(archive), "manifest": prepare_manifest(pkg)},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"[ROSClaw] Packaged: {archive}")
    return 0


def cmd_skill_verify_package(args: argparse.Namespace) -> int:
    result = verify_package(Path(args.archive).expanduser().resolve())
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Verify package: {'PASS' if result['ok'] else 'FAIL'}")
        for e in result["errors"]:
            print(f"  ✗ {e}")
        for w in result["warnings"]:
            print(f"  ! {w}")
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def cmd_skill_upload(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()

    try:
        result = upload_skill(
            pkg,
            visibility=args.visibility,
            hub_base_url=args.hub_base_url,
            api_key_env=args.api_key_env,
            dry_run=args.dry_run,
            force=args.force,
        )
    except RuntimeError as exc:
        print(f"[ROSClaw] Upload failed: {exc}")
        return 1

    if args.json:
        # Mask API key in payload if present.
        payload = result.get("payload", {})
        print(
            json.dumps(
                {
                    "ok": result["ok"],
                    "dry_run": result["dry_run"],
                    "payload": payload,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"[ROSClaw] Upload: {'DRY-RUN' if result['dry_run'] else 'OK'}")
        print(f"  skill: {result['payload']['name']}")
        print(f"  version: {result['payload']['version']}")
        print(f"  visibility: {args.visibility}")
    return 0


# ---------------------------------------------------------------------------
# Submit catalog
# ---------------------------------------------------------------------------


def cmd_skill_submit_catalog(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()

    try:
        result = submit_to_catalog(
            pkg,
            dry_run=args.dry_run,
            catalog_repo=args.catalog_repo,
            base_branch=args.base_branch,
            branch_prefix=args.branch_prefix,
        )
    except RuntimeError as exc:
        print(f"[ROSClaw] Submit to catalog failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if result["dry_run"]:
            print("[ROSClaw] Submit to catalog: DRY-RUN")
            print(f"  skill: {result['skill_name']}")
            print(f"  version: {result['version']}")
            print(f"  target repo: {result['catalog_repo']}")
            print(f"  base branch: {result['base_branch']}")
            print(f"  proposed branch: {result['branch']}")
        else:
            print("[ROSClaw] Submit to catalog: PR created")
            print(f"  skill: {result['skill_name']}")
            print(f"  version: {result['version']}")
            print(f"  flow: {result.get('flow', 'fork')}")
            print(f"  PR: {result.get('pr_url', 'unknown')}")
    return 0


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def cmd_skill_rollback(args: argparse.Namespace) -> int:
    skill_dir = _load_skill_dir_arg(args)
    if not skill_dir.exists():
        print(f"[ROSClaw] Skill not found: {skill_dir}")
        return 1
    pkg = SkillPackage(skill_dir).try_load()
    try:
        result = rollback_skill(pkg, to_version=args.to, reason=args.reason or "")
    except ValueError as exc:
        print(f"[ROSClaw] Rollback failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Rolled back to v{result['to_version']}")
        print(f"  evidence: {result['evidence']}")
    return 0


def cmd_skill_search(args: argparse.Namespace) -> int:
    """Search the unified catalog, or (no query) list builtin + local skills.

    Skill Runtime 2.0 (doc §10): with a query this searches across
    builtin/installed/official/workspace sources — the official
    ``ros-claw/skills`` registry included (fetched once, then cached).
    """
    query = getattr(args, "query", None)
    if query:
        return _cmd_skill_search_catalog(args, query)
    builtins = list_builtin_skills()
    registry = SkillLocalRegistry()
    local = registry.list_skills()
    if args.json:
        print(json.dumps({"builtin": builtins, "local": local}, indent=2, ensure_ascii=False))
        return 0
    print("[ROSClaw] Builtin skills")
    for s in builtins:
        print(f"  {s['name']:<30} {s.get('display_name', '')}")
    print("[ROSClaw] Local skill-hub packages")
    for s in local:
        print(f"  {s.get('name', 'unknown')}")
    return 0


def _cmd_skill_search_catalog(args: argparse.Namespace, query: str) -> int:
    from rosclaw.skill.catalog_service import SkillCatalogService

    hits = SkillCatalogService.default().search(query)
    if args.json:
        print(
            json.dumps(
                {"results": [h.to_dict() for h in hits]}, indent=2, ensure_ascii=False
            )
        )
        return 0
    print(f'[ROSClaw] Skill catalog results for "{query}"')
    if not hits:
        print("  (no matching skills)")
        return 0
    for h in hits:
        badges = []
        if h.official:
            badges.append("official")
        if h.installed:
            badges.append("installed")
        badge = f"  [{', '.join(badges)}]" if badges else ""
        version = f"@{h.version}" if h.version else ""
        print(f"  {h.name}{version}{badge}")
        if h.description:
            print(f"      {h.description}")
        status = h.verification_status or "unverified"
        print(f"      source={h.source} installed={'yes' if h.installed else 'no'} status={status}")
    return 0


def cmd_skill_install(args: argparse.Namespace) -> int:
    """Install a skill: namespaced refs install from catalogs (doc §11),
    bare names keep the legacy builtin-registration behavior.
    """
    name = args.name
    if "/" in name:
        return _cmd_skill_install_remote(args, name)
    entry = get_builtin_skill(name)
    if entry is None:
        print(f"[ROSClaw] Builtin skill not found: {name}")
        print("[ROSClaw] Run `rosclaw skill search` to list available skills")
        return 1

    registry = SkillLocalRegistry()
    data = {
        "local_path": str(Path(__file__).parent / "builtins" / name),
        "current_version": entry.version,
        "current_stage": "installable",
        "last_eval_report": None,
        "builtin": True,
    }
    registry._data["skills"][name] = data
    registry._save()
    print(f"[ROSClaw] Installed builtin skill: {name}@{entry.version}")
    return 0


def _cmd_skill_install_remote(args: argparse.Namespace, ref: str) -> int:
    from rosclaw.skill.installer import SkillInstaller, SkillInstallError

    try:
        receipt = SkillInstaller().install(ref)
    except SkillInstallError as exc:
        print(f"[ROSClaw] Install failed: {exc}")
        return 1
    if args.json:
        print(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Installed {receipt.name}@{receipt.version}")
        print(f"  digest: {receipt.package_digest}")
        print(f"  trust:  {receipt.trust}")
        print(f"  path:   {receipt.install_dir}")
    return 0


def cmd_skill_run(args: argparse.Namespace) -> int:
    """Run an installed host skill action (doc §16).

    ``--action plan`` loads the skill's planner entrypoint, computes a
    typed ExecutionPlan from the detected host state, validates it
    against HostOps policy and prints it (with its plan hash).
    ``--action install`` requires ``--approve <plan_hash>`` matching the
    exact plan (doc §21) and then enters the local-TTY authorization
    flow (doc §23) — the agent never sees credentials.
    """
    ref = args.name
    if "/" not in ref:
        print(f"[ROSClaw] `skill run` expects a namespaced ref, got: {ref}")
        return 1
    try:
        plan = _build_skill_plan(ref, args)
    except _SkillRunError as exc:
        print(f"[ROSClaw] {exc}")
        return 1

    from rosclaw.hostops.planner import plan_hash
    from rosclaw.hostops.policy import HostOpsPolicy, HostOpsPolicyError

    policy = HostOpsPolicy()
    try:
        policy.validate_plan(plan)
    except HostOpsPolicyError as exc:
        print(f"[ROSClaw] Plan rejected by HostOps policy: {exc}")
        return 1
    plan["plan_hash"] = plan_hash(plan)

    if args.action == "plan":
        if args.json:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        else:
            print(f"[ROSClaw] Execution plan for {plan['skill']}")
            print(f"  domain: {plan['domain']}  plan_hash: {plan['plan_hash'][:16]}…")
            for op in plan["operations"]:
                print(f"  - {op['type']}")
            print("[ROSClaw] Approve with: rosclaw skill run "
                  f"{ref} --action install --approve {plan['plan_hash']}")
        return 0

    # --action install: approval bound to the exact plan hash (doc §21).
    if not args.approve or args.approve != plan["plan_hash"]:
        print("[ROSClaw] Execution requires an approval bound to this plan.")
        print(f"  plan_hash: {plan['plan_hash']}")
        print(f"  approve with: rosclaw skill run {ref} --action install "
              f"--approve {plan['plan_hash']}")
        return 2

    approval = policy.approve(plan["plan_hash"])
    try:
        policy.require_approval(plan, approval)
    except HostOpsPolicyError as exc:
        print(f"[ROSClaw] {exc}")
        return 2

    from rosclaw.hostops.auth import begin_local_authorization
    from rosclaw.hostops.receipt import new_job_id

    auth_request = begin_local_authorization(new_job_id())
    if args.json:
        print(json.dumps({**auth_request, "plan": plan}, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] {auth_request['status']}: {auth_request['instruction']}")
    return 0


class _SkillRunError(Exception):
    pass


def _build_skill_plan(ref: str, args: argparse.Namespace) -> dict:
    """Load the installed skill's planner and produce the ExecutionPlan."""
    import importlib.util

    import yaml

    from rosclaw.firstboot.workspace import get_rosclaw_home
    from rosclaw.hostops.models import make_plan
    from rosclaw.skill.resolver import detect_host_context

    home = get_rosclaw_home()
    lockfile = home / "skills" / "installed.lock.json"
    installed = {}
    if lockfile.exists():
        try:
            installed = json.loads(lockfile.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            installed = {}
    if ref not in installed:
        raise _SkillRunError(
            f"skill {ref} is not installed; run `rosclaw skill install {ref}` first"
        )
    version = str(installed[ref].get("version", ""))
    install_dir = home / "skills" / ref / version
    manifest_path = install_dir / "skill.yaml"
    if not manifest_path.exists():
        raise _SkillRunError(f"installed skill {ref}@{version} has no skill.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    execution = manifest.get("execution", {}) or {}
    planner_spec = (execution.get("planner", {}) or {}).get("entrypoint")
    if not planner_spec:
        raise _SkillRunError(f"skill {ref} declares no execution.planner entrypoint")
    module_name, _, func_name = planner_spec.partition(":")
    if not module_name or not func_name:
        raise _SkillRunError(f"invalid planner entrypoint {planner_spec!r}")
    entrypoint_path = install_dir / module_name
    if not entrypoint_path.exists():
        raise _SkillRunError(f"planner module {module_name} missing in {install_dir}")

    # Import the skill's planner. Trust basis: the package was digest-pinned
    # at install time; its *output* is policy-checked before anything runs.
    spec = importlib.util.spec_from_file_location(f"rosclaw_skill_{ref}", entrypoint_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    planner_fn = getattr(module, func_name, None)
    if not callable(planner_fn):
        raise _SkillRunError(f"planner {planner_spec} is not callable")

    context = detect_host_context()
    skill_args = json.loads(args.args) if getattr(args, "args", None) else {}
    raw_plan = planner_fn(context, skill_args)
    if not isinstance(raw_plan, dict) or not isinstance(raw_plan.get("operations"), list):
        raise _SkillRunError("planner must return a dict with an operations list")

    host_target = {
        "os": context.get("os", ""),
        "version": context.get("os_version", ""),
        "arch": context.get("arch", ""),
    }
    return make_plan(
        skill=raw_plan.get("skill") or f"{ref}@{version}",
        domain=raw_plan.get("domain") or execution.get("domain", "host"),
        target=raw_plan.get("target") or host_target,
        operations=raw_plan["operations"],
    )


def cmd_skill_inspect(args: argparse.Namespace) -> int:
    """Show details for a builtin or local skill."""
    name = args.name
    entry = get_builtin_skill(name)
    if entry is not None:
        info = {
            "name": entry.name,
            "description": entry.description,
            "version": entry.version,
            "skill_type": entry.skill_type,
            "requirements": entry.requirements,
            "metadata": entry.metadata,
            "builtin": True,
        }
    else:
        registry = SkillLocalRegistry()
        local = {s.get("name"): s for s in registry.list_skills()}
        if name not in local:
            print(f"[ROSClaw] Skill not found: {name}")
            return 1
        info = {"name": name, "local": local[name]}
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    else:
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    return 0


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def add_skill_hub_parsers(skill_subparsers: Any) -> None:
    search_parser = skill_subparsers.add_parser(
        "search",
        help="Search skills across builtin/installed/official/workspace catalogs",
    )
    search_parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Intent or keywords (e.g. \"install ros2\"); omit to list builtin+local",
    )
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")
    search_parser.set_defaults(func=cmd_skill_search)

    install_parser = skill_subparsers.add_parser(
        "install", help="Install a builtin skill reference"
    )
    install_parser.add_argument("name", help="Skill name")
    install_parser.add_argument("--json", action="store_true", help="Output as JSON")
    install_parser.set_defaults(func=cmd_skill_install)

    run_parser = skill_subparsers.add_parser(
        "run", help="Run an installed skill action (plan/install)"
    )
    run_parser.add_argument("name", help="Namespaced skill ref (e.g. ros-claw/ros_install)")
    run_parser.add_argument(
        "--action",
        choices=["plan", "install"],
        default="plan",
        help="plan: compute and validate the ExecutionPlan; install: execute it",
    )
    run_parser.add_argument(
        "--approve",
        default=None,
        help="Approval token: the plan_hash shown by --action plan",
    )
    run_parser.add_argument(
        "--args",
        default=None,
        help="JSON object passed to the skill planner as args",
    )
    run_parser.add_argument("--json", action="store_true", help="Output as JSON")
    run_parser.set_defaults(func=cmd_skill_run)

    inspect_parser = skill_subparsers.add_parser("inspect", help="Inspect a skill")
    inspect_parser.add_argument("name", help="Skill name")
    inspect_parser.add_argument("--json", action="store_true", help="Output as JSON")
    inspect_parser.set_defaults(func=cmd_skill_inspect)

    init_parser = skill_subparsers.add_parser("init", help="Create a local skill package skeleton")
    init_parser.add_argument("name", help="Skill name")
    init_parser.add_argument("--robot", default=None, help="Default robot")
    init_parser.add_argument("--category", default=None, help="Skill category")
    init_parser.add_argument("--namespace", default=None, help="Skill namespace")
    init_parser.add_argument("--template", default=None, help="Template name")
    init_parser.add_argument("--output", default=None, help="Output directory")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing")
    init_parser.set_defaults(func=cmd_skill_init)

    validate_parser = skill_subparsers.add_parser("validate", help="Validate skill package")
    validate_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    validate_parser.add_argument("--name", default=None, help="Skill name (used to resolve dir)")
    validate_parser.add_argument("--workspace", default=None, help="Workspace root")
    validate_parser.add_argument("--json", action="store_true", help="Output JSON")
    validate_parser.set_defaults(func=cmd_skill_validate)

    mine_parser = skill_subparsers.add_parser(
        "mine", help="Mine skill candidate from practice episodes"
    )
    mine_parser.add_argument(
        "--from", dest="source", required=True, help="Practice episodes directory"
    )
    mine_parser.add_argument("--task", required=True, help="Task name")
    mine_parser.add_argument("--robot", default=None, help="Robot filter")
    mine_parser.add_argument("--output", default=None, help="Skill output directory")
    mine_parser.add_argument("--candidate", default=None, help="Candidate ID")
    mine_parser.add_argument("--json", action="store_true", help="Output JSON")
    mine_parser.set_defaults(func=cmd_skill_mine)

    eval_parser = skill_subparsers.add_parser("eval", help="Evaluate skill candidate")
    eval_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    eval_parser.add_argument("--name", default=None, help="Skill name")
    eval_parser.add_argument("--candidate", default=None, help="Candidate ID")
    eval_parser.add_argument(
        "--mode", default="replay", choices=["replay", "sandbox"], help="Eval mode"
    )
    eval_parser.add_argument(
        "--save-evidence", action="store_true", default=True, help="Write eval report"
    )
    eval_parser.add_argument("--json", action="store_true", help="Output JSON")
    eval_parser.set_defaults(func=cmd_skill_eval)

    promote_parser = skill_subparsers.add_parser("promote", help="Promote candidate to version")
    promote_parser.add_argument("skill_ref", help="Skill ref: name@candidate_id")
    promote_parser.add_argument("--to-version", required=True, help="Target version")
    promote_parser.add_argument("--stage", default="validated", help="Target stage")
    promote_parser.add_argument(
        "--require-eval-pass", action="store_true", default=True, help="Require eval pass"
    )
    promote_parser.add_argument("--workspace", default=None, help="Workspace root")
    promote_parser.add_argument("--json", action="store_true", help="Output JSON")
    promote_parser.set_defaults(func=cmd_skill_promote)

    package_parser = skill_subparsers.add_parser(
        "package", help="Package skill into distributable archive"
    )
    package_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    package_parser.add_argument("--name", default=None, help="Skill name")
    package_parser.add_argument("--output", default="dist", help="Output directory")
    package_parser.add_argument(
        "--format", default="tar.gz", choices=["tar.gz"], help="Archive format"
    )
    package_parser.add_argument(
        "--include-evidence",
        default="summary",
        choices=["none", "summary", "full"],
        help="Evidence inclusion",
    )
    package_parser.add_argument("--workspace", default=None, help="Workspace root")
    package_parser.add_argument("--json", action="store_true", help="Output JSON")
    package_parser.set_defaults(func=cmd_skill_package)

    verify_pkg_parser = skill_subparsers.add_parser(
        "verify-package", help="Verify packaged archive"
    )
    verify_pkg_parser.add_argument("archive", help="Archive path")
    verify_pkg_parser.add_argument("--json", action="store_true", help="Output JSON")
    verify_pkg_parser.set_defaults(func=cmd_skill_verify_package)

    upload_parser = skill_subparsers.add_parser(
        "upload", help="Upload skill metadata to ROSClaw Hub (admin only)"
    )
    upload_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    upload_parser.add_argument("--name", default=None, help="Skill name")
    upload_parser.add_argument(
        "--visibility",
        default="private",
        choices=["public", "private", "org", "unlisted"],
        help="Visibility",
    )
    upload_parser.add_argument(
        "--hub-base-url", default="https://www.rosclaw.io", help="Hub base URL"
    )
    upload_parser.add_argument(
        "--api-key-env", default="ROSCLAW_ADMIN_API_KEY", help="API key env var"
    )
    upload_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    upload_parser.add_argument("--force", action="store_true", help="Force update on conflict")
    upload_parser.add_argument("--workspace", default=None, help="Workspace root")
    upload_parser.add_argument("--json", action="store_true", help="Output JSON")
    upload_parser.set_defaults(func=cmd_skill_upload)

    submit_catalog_parser = skill_subparsers.add_parser(
        "submit-catalog",
        help="Submit a local skill to the official ros-claw/skills catalog via GitHub PR",
    )
    submit_catalog_parser.add_argument("skill_dir", nargs="?", help="Skill directory")
    submit_catalog_parser.add_argument("--name", default=None, help="Skill name")
    submit_catalog_parser.add_argument(
        "--catalog-repo", default="ros-claw/skills", help="Upstream catalog repo"
    )
    submit_catalog_parser.add_argument(
        "--base-branch", default="main", help="Base branch in upstream repo"
    )
    submit_catalog_parser.add_argument(
        "--branch-prefix", default="add", help="Feature branch prefix"
    )
    submit_catalog_parser.add_argument("--workspace", default=None, help="Workspace root")
    submit_catalog_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    submit_catalog_parser.add_argument("--json", action="store_true", help="Output JSON")
    submit_catalog_parser.set_defaults(func=cmd_skill_submit_catalog)
