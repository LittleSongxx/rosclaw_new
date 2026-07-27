"""Tests that the imperative demo apps coexist with the declarative Capability Apps.

`rosclaw.app` (singular) holds upstream's declarative Capability App verbs;
`rosclaw.apps` (plural) holds our imperative ROS demo apps. Both register into
the single `rosclaw app` subparser via the shared `set_defaults(app_handler=...)`
convention. These tests pin that contract: the two families must not shadow each
other, and every registered command must carry a dispatchable handler.
"""

from __future__ import annotations

import argparse
from typing import Any

from rosclaw.app.cli import add_app_subparsers, dispatch_app_command
from rosclaw.apps.cli import add_demo_app_subparsers

# Upstream's declarative Capability App verbs.
DECLARATIVE_COMMANDS = {"install", "list", "init", "add", "validate", "run"}

# Our imperative demo-app verbs.
DEMO_COMMANDS = {
    "move",
    "patrol",
    "nav2-check",
    "nav2-go",
    "nav2-demo",
    "cmu-check",
    "cmu-launch",
    "cmu-go",
    "cmu-explore",
    "cmu-chat",
    "cmu-dashboard",
    "cmu-demo",
}


def _build_app_parser() -> tuple[argparse.ArgumentParser, Any]:
    """Build a parser wired the same way `rosclaw.cli.main` wires it."""
    parser = argparse.ArgumentParser(prog="rosclaw")
    app = parser.add_subparsers(dest="command").add_parser("app")
    commands = app.add_subparsers(dest="app_command")
    add_app_subparsers(commands)
    # Registered after the declarative verbs so those keep priority on collision.
    add_demo_app_subparsers(commands)
    return parser, commands


def _registered_commands() -> dict[str, argparse.ArgumentParser]:
    _, commands = _build_app_parser()
    return dict(commands.choices)


def test_both_app_families_register_into_one_namespace():
    registered = set(_registered_commands())

    assert DECLARATIVE_COMMANDS <= registered
    assert DEMO_COMMANDS <= registered


def test_the_two_families_do_not_collide():
    """A name in both families would mean one silently shadows the other."""

    assert DECLARATIVE_COMMANDS & DEMO_COMMANDS == set()


def test_demo_registration_does_not_displace_declarative_verbs():
    """Registering ours second must leave upstream's parsers in place."""

    declarative_only = argparse.ArgumentParser(prog="rosclaw")
    only_commands = (
        declarative_only.add_subparsers(dest="command")
        .add_parser("app")
        .add_subparsers(dest="app_command")
    )
    add_app_subparsers(only_commands)
    before = dict(only_commands.choices)

    add_demo_app_subparsers(only_commands)

    for name, parser in before.items():
        assert only_commands.choices[name] is parser, name


def test_every_registered_command_has_a_handler():
    """`dispatch_app_command` returns 1 for any command missing app_handler."""

    _, commands = _build_app_parser()

    for name, subparser in commands.choices.items():
        handler = subparser.get_default("app_handler")
        assert callable(handler), f"{name} has no dispatchable app_handler"


def test_demo_commands_parse_and_bind_their_handler():
    parser, _ = _build_app_parser()

    args = parser.parse_args(["app", "move", "前进 1 米"])

    assert args.command == "app"
    assert args.app_command == "move"
    assert callable(args.app_handler)


def test_declarative_commands_still_parse():
    parser, _ = _build_app_parser()

    args = parser.parse_args(["app", "list"])

    assert args.app_command == "list"
    assert callable(args.app_handler)


def test_dispatch_routes_to_the_bound_handler():
    parser, _ = _build_app_parser()
    args = parser.parse_args(["app", "move", "前进 1 米"])

    calls: list[argparse.Namespace] = []

    def _fake_handler(parsed: argparse.Namespace) -> int:
        calls.append(parsed)
        return 0

    args.app_handler = _fake_handler

    assert dispatch_app_command(args) == 0
    assert len(calls) == 1


def test_cmu_dashboard_defaults_to_loopback():
    """The dashboard has no authentication, so it must not default to 0.0.0.0."""

    parser, _ = _build_app_parser()

    args = parser.parse_args(["app", "cmu-dashboard"])

    assert args.host == "127.0.0.1"
