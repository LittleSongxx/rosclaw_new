"""Local authorization for privileged host operations (doc §23).

The sudo password never enters the agent context — not the LLM, not MCP,
not traces, not memory. The agent only ever sees
``AUTHENTICATION_REQUIRED`` plus an instruction for the operator to run
on a local TTY. Dashboard/operator auth UX is a follow-up.
"""

from __future__ import annotations


def begin_local_authorization(job_id: str) -> dict:
    """Start the local-TTY authorization flow for a job.

    Deliberately takes no credential material: authentication happens
    between the operator and their own machine, outside any agent-visible
    channel.
    """
    return {
        "status": "AUTHENTICATION_REQUIRED",
        "job_id": job_id,
        "channel": "local_tty",
        "instruction": (
            f"run `rosclaw host authorize {job_id}` on the host to "
            f"authenticate with sudo locally"
        ),
    }
