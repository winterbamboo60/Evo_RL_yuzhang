"""Wire-compatible alias of LeRobot's LearnerService implementation."""

from lerobot.rl.learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService

__all__ = ["MAX_WORKERS", "SHUTDOWN_TIMEOUT", "LearnerService"]
