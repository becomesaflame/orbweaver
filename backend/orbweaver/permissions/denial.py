"""Session-scoped classifier denial counters."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

DENIAL_LIMITS = {
    "max_consecutive": 3,
    "max_total": 20,
}


@dataclass
class DenialTrackingState:
    consecutive_denials: int = 0
    total_denials: int = 0

    def record_denial(self) -> None:
        self.consecutive_denials += 1
        self.total_denials += 1

    def record_success(self) -> None:
        self.consecutive_denials = 0

    def should_fallback(self) -> bool:
        return (
            self.consecutive_denials >= DENIAL_LIMITS["max_consecutive"]
            or self.total_denials >= DENIAL_LIMITS["max_total"]
        )


_states: dict[UUID, DenialTrackingState] = {}


def denial_state_for(session_id: UUID) -> DenialTrackingState:
    state = _states.get(session_id)
    if state is None:
        state = DenialTrackingState()
        _states[session_id] = state
    return state


def reset_denial_states() -> None:
    _states.clear()
