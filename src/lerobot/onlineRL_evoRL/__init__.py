"""PI05-RLT online RL entry points and Piper-specific runtime.

This package owns the actor/learner lifecycles, compact episode schema,
head-only checkpointing, GPU handoff, robot-facing controls and preflight.
Only low-level replay, trainer and byte-transport primitives are shared.
"""

__all__: list[str] = []
