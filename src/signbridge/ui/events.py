"""Module 15: UI Communication & Typed Event Bus."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeVar


@dataclass(frozen=True)
class BaseEvent:
    timestamp: float


@dataclass(frozen=True)
class SignDetectedEvent(BaseEvent):
    sign_label: str
    confidence: float
    is_stable: bool


@dataclass(frozen=True)
class WordUpdatedEvent(BaseEvent):
    current_word: str
    committed_word: str | None


@dataclass(frozen=True)
class SentenceUpdatedEvent(BaseEvent):
    sentence_text: str


T = TypeVar("T", bound=BaseEvent)


class EventBus:
    """Decoupled publish-subscribe event dispatcher."""

    def __init__(self) -> None:
        self._subscribers: dict[Type[BaseEvent], list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: Type[T], callback: Callable[[T], None]) -> None:
        """Registers a listener callback for an event type."""
        self._subscribers[event_type].append(callback)

    def publish(self, event: BaseEvent) -> None:
        """Dispatches event to all registered listeners."""
        for callback in self._subscribers[type(event)]:
            callback(event)
