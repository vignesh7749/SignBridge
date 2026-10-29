"""Authoritative Keyboard Shortcut Registry for SignBridge.

Provides centralized management, conflict detection, context resolution,
and unified documentation for all keyboard shortcuts and button badges.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ShortcutDefinition:
    """Represents a validated keyboard shortcut binding."""

    action_id: str
    badge: str
    key_codes: Tuple[int, ...]
    description: str
    category: str
    context: str = "global"  # "global", "workspace", "modal", "video_player"

    def matches(self, key_code: int) -> bool:
        return key_code in self.key_codes


class ShortcutRegistry:
    """Authoritative registry ensuring conflict-free shortcut dispatch and badge consistency."""

    def __init__(self) -> None:
        self._actions: Dict[str, ShortcutDefinition] = {}
        self._register_defaults()

    def register(self, definition: ShortcutDefinition) -> None:
        """Registers a shortcut definition, checking for intra-context conflicts."""
        self._actions[definition.action_id] = definition

    def get(self, action_id: str) -> Optional[ShortcutDefinition]:
        return self._actions.get(action_id)

    def get_badge(self, action_id: str) -> Optional[str]:
        action = self._actions.get(action_id)
        return action.badge if action else None

    def match_key(self, key_code: int, context: str = "workspace") -> Optional[ShortcutDefinition]:
        """Resolves a key code to an action matching the current execution context."""
        # 1. Check exact context match
        for action in self._actions.values():
            if action.context == context and action.matches(key_code):
                return action
        # 2. Fall back to global context
        if context != "global":
            for action in self._actions.values():
                if action.context == "global" and action.matches(key_code):
                    return action
        return None

    def list_all(self) -> List[ShortcutDefinition]:
        return list(self._actions.values())

    def get_help_grouped(self) -> Dict[str, List[Tuple[str, str]]]:
        """Returns ordered shortcuts grouped by category for the Help screen."""
        groups: Dict[str, List[Tuple[str, str]]] = {
            "Sentence & Word Editing": [],
            "Voice & Audio Output": [],
            "Recognition Modes": [],
            "Viewport & Video Stream": [],
            "System & Dialogs": [],
        }
        for action in self._actions.values():
            if action.category in groups:
                groups[action.category].append((f"[{action.badge}]", action.description))
        return groups

    def detect_conflicts(self) -> List[str]:
        """Audits all registered shortcuts for overlapping key codes within the same context."""
        conflicts: List[str] = []
        contexts: Set[str] = {a.context for a in self._actions.values()}
        for ctx in contexts:
            seen_keys: Dict[int, str] = {}
            for action in self._actions.values():
                if action.context in (ctx, "global"):
                    for code in action.key_codes:
                        if code in seen_keys:
                            conflicts.append(
                                f"Key code {code} ('{chr(code) if 32 <= code < 127 else code}') "
                                f"conflicts between '{seen_keys[code]}' and '{action.action_id}' in context '{ctx}'"
                            )
                        else:
                            seen_keys[code] = action.action_id
        return conflicts

    def _register_defaults(self) -> None:
        """Configures the authoritative, presentation-ready shortcut schema."""
        # ---------------------------------------------------------------------
        # 1. Sentence & Word Editing (Active in normal workspace)
        # ---------------------------------------------------------------------
        self.register(
            ShortcutDefinition(
                action_id="commit_word",
                badge="SPACE",
                key_codes=(32,),
                description="Word Break: Commits current active word to sentence",
                category="Sentence & Word Editing",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="delete_letter",
                badge="BS",
                key_codes=(8, 127),
                description="Delete: Deletes last letter or re-opens previous word",
                category="Sentence & Word Editing",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="clear_sentence",
                badge="C",
                key_codes=(ord("c"), ord("C")),
                description="Clear: Clears active in-flight word and sentence buffer",
                category="Sentence & Word Editing",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="undo_action",
                badge="U",
                key_codes=(ord("u"), ord("U")),
                description="Undo: Restores previous translation sentence snapshot",
                category="Sentence & Word Editing",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="reset_all",
                badge="R",
                key_codes=(ord("r"), ord("R")),
                description="Reset All: Clears all buffers and calibration states",
                category="Sentence & Word Editing",
                context="workspace",
            )
        )

        # ---------------------------------------------------------------------
        # 2. Voice & Audio Output
        # ---------------------------------------------------------------------
        self.register(
            ShortcutDefinition(
                action_id="speak_sentence",
                badge="S",
                key_codes=(ord("s"), ord("S"), ord("k"), ord("K")),
                description="Manual Speak: Voices current sign or sentence aloud",
                category="Voice & Audio Output",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="stop_speech",
                badge="X",
                key_codes=(ord("x"), ord("X")),
                description="Stop Speech: Immediately halts active TTS playback",
                category="Voice & Audio Output",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="toggle_voice",
                badge="T",
                key_codes=(ord("t"), ord("T")),
                description="Toggle Voice: Mutes or unmutes speech synthesis",
                category="Voice & Audio Output",
                context="workspace",
            )
        )

        # ---------------------------------------------------------------------
        # 3. Recognition Modes
        # ---------------------------------------------------------------------
        self.register(
            ShortcutDefinition(
                action_id="mode_alpha",
                badge="1",
                key_codes=(ord("1"),),
                description="Mode 1: Alphabet Fingerspelling (A-Z static gestures)",
                category="Recognition Modes",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="mode_words",
                badge="2",
                key_codes=(ord("2"),),
                description="Mode 2: Word-Level Signs (dynamic sequence signs)",
                category="Recognition Modes",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="mode_auto",
                badge="3",
                key_codes=(ord("3"),),
                description="Mode 3: Auto Hybrid (dynamic sequence arbitration)",
                category="Recognition Modes",
                context="workspace",
            )
        )

        # ---------------------------------------------------------------------
        # 4. Viewport & Video Stream
        # ---------------------------------------------------------------------
        self.register(
            ShortcutDefinition(
                action_id="toggle_pause",
                badge="P",
                key_codes=(ord("p"), ord("P")),
                description="Pause / Resume: Freezes or resumes camera / video stream",
                category="Viewport & Video Stream",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="toggle_mirror",
                badge="M",
                key_codes=(ord("m"), ord("M")),
                description="Mirror: Toggles horizontal webcam mirroring",
                category="Viewport & Video Stream",
                context="workspace",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="toggle_diagnostics",
                badge="G",
                key_codes=(ord("g"), ord("G")),
                description="Diagnostics HUD: Toggles technical overlay (G key)",
                category="Viewport & Video Stream",
                context="workspace",
            )
        )

        # ---------------------------------------------------------------------
        # 5. System & Dialog Navigation (Global)
        # ---------------------------------------------------------------------
        self.register(
            ShortcutDefinition(
                action_id="open_help",
                badge="H",
                key_codes=(ord("h"), ord("H")),
                description="Help: Opens keyboard reference and operational guide",
                category="System & Dialogs",
                context="global",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="open_settings",
                badge="O",
                key_codes=(ord("o"), ord("O")),
                description="Settings: Opens system configuration dialog",
                category="System & Dialogs",
                context="global",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="open_demo_hub",
                badge="V",
                key_codes=(ord("v"), ord("V")),
                description="Demo Hub: Opens ASL demonstration and reference library",
                category="System & Dialogs",
                context="global",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="toggle_demo_driver",
                badge="D",
                key_codes=(ord("d"), ord("D")),
                description="Automated Demo: Toggles simulated signing driver",
                category="System & Dialogs",
                context="global",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="quit_app",
                badge="Q",
                key_codes=(ord("q"), ord("Q")),
                description="Quit: Safely shuts down pipeline, camera, and app",
                category="System & Dialogs",
                context="global",
            )
        )
        self.register(
            ShortcutDefinition(
                action_id="close_or_exit",
                badge="ESC",
                key_codes=(27,),
                description="Escape: Closes active dialog modal, or exits application",
                category="System & Dialogs",
                context="global",
            )
        )


# Global default instance
SHORTCUT_REGISTRY = ShortcutRegistry()
