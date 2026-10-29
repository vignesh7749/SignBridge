"""Sign Language Reference Library & Demonstration Catalog.

Provides structured access, searching, category filtering, pagination, and
asset resolution for the comprehensive ASL reference library and demo hub.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional, Sequence, Union

from signbridge.core.logger import get_logger

logger = get_logger("video.catalog")


@dataclass(frozen=True)
class CatalogSignEntry:
    """Individual sign entry in the reference catalog."""

    id: str
    name: str
    category: str
    type: str  # 'letter', 'number', 'word', 'phrase'
    description: str
    language: str = "ASL"
    video: Optional[str] = None
    available: bool = False
    source: str = "ASL Reference Library"
    license: str = "Educational Reference"
    notes: str = ""

    def get_full_video_path(self, project_root: Optional[Path] = None) -> Optional[Path]:
        """Resolves the absolute path to the demonstration video if available."""
        if not self.video:
            return None
        root = project_root or Path(__file__).resolve().parents[3]
        candidate = (root / self.video).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    def matches_query(self, query: str) -> bool:
        """Case-insensitive substring search matching name, description, category, or notes."""
        if not query or not query.strip():
            return True
        q = query.strip().lower()
        return (
            q in self.name.lower()
            or q in self.description.lower()
            or q in self.category.lower()
            or q in self.notes.lower()
            or q in self.type.lower()
        )

    def to_dict(self) -> dict:
        return asdict(self)


class SignLanguageCatalog:
    """Central manager for loading, querying, and navigating sign language references."""

    DEFAULT_CATEGORIES: list[str] = [
        "All Signs",
        "Alphabet",
        "Numbers",
        "Greetings",
        "Common Words",
        "Daily Communication",
        "Common Phrases",
    ]

    def __init__(
        self,
        catalog_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.project_root = Path(
            project_root or Path(__file__).resolve().parents[3]
        ).resolve()
        self.catalog_path = Path(
            catalog_path or (self.project_root / "datasets" / "sign_language_catalog.json")
        ).resolve()

        self._entries: list[CatalogSignEntry] = []
        self._categories: list[str] = list(self.DEFAULT_CATEGORIES)
        self._id_index: dict[str, CatalogSignEntry] = {}
        self.load()

    def load(self) -> None:
        """Loads and parses the catalog JSON file. Falls back gracefully if absent."""
        self._entries.clear()
        self._id_index.clear()

        if not self.catalog_path.exists():
            logger.warning("Sign language catalog file not found: %s", self.catalog_path)
            self._load_fallback_catalog()
            return

        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_categories = data.get("categories", [])
            cats = ["All Signs"]
            for c in raw_categories:
                if c not in cats:
                    cats.append(c)
            self._categories = cats

            for item in data.get("signs", []):
                v_rel = item.get("video")
                # Verify physical existence if marked available or has video path
                physically_available = False
                if v_rel:
                    p = (self.project_root / v_rel).resolve()
                    physically_available = p.exists() and p.is_file()

                entry = CatalogSignEntry(
                    id=item.get("id", ""),
                    name=item.get("name", "Unknown"),
                    language=item.get("language", "ASL"),
                    category=item.get("category", "General"),
                    type=item.get("type", "word"),
                    description=item.get("description", ""),
                    video=v_rel if physically_available else item.get("video"),
                    available=physically_available or bool(item.get("available", False)),
                    source=item.get("source", "ASL Reference Library"),
                    license=item.get("license", "Educational Reference"),
                    notes=item.get("notes", ""),
                )
                if entry.id:
                    self._entries.append(entry)
                    self._id_index[entry.id] = entry

            logger.info(
                "Loaded %d signs (%d with video) across %d categories from %s",
                len(self._entries),
                sum(1 for e in self._entries if e.available),
                len(self._categories),
                self.catalog_path,
            )
        except Exception as e:
            logger.error("Failed to load sign language catalog %s: %s", self.catalog_path, e)
            self._load_fallback_catalog()

    def _load_fallback_catalog(self) -> None:
        """Loads minimal built-in word signs when catalog json is unavailable."""
        fallback_signs = [
            ("asl_word_hello", "HELLO", "Greetings", "word", "Temple wave salute (1-hand)", "datasets/samples/word_signs/hello.mp4"),
            ("asl_word_help", "HELP", "Common Words", "word", "Thumbs-up lifted on flat palm (2-hand)", "datasets/samples/word_signs/help.mp4"),
            ("asl_word_yes", "YES", "Common Words", "word", "S-fist vertical nodding (1-hand)", "datasets/samples/word_signs/yes.mp4"),
            ("asl_word_no", "NO", "Common Words", "word", "Index and middle snap shut (1-hand)", "datasets/samples/word_signs/no.mp4"),
            ("asl_word_stop", "STOP", "Common Words", "word", "Open palm chop onto base (2-hand)", "datasets/samples/word_signs/stop.mp4"),
            ("asl_word_thank_you", "THANK YOU", "Greetings", "word", "Chin touch extending forward (1-hand)", "datasets/samples/word_signs/thank_you.mp4"),
        ]
        for cid, name, cat, stype, desc, vpath in fallback_signs:
            p = (self.project_root / vpath).resolve()
            avail = p.exists() and p.is_file()
            entry = CatalogSignEntry(
                id=cid,
                name=name,
                language="ASL",
                category=cat,
                type=stype,
                description=desc,
                video=vpath if avail else None,
                available=avail,
                source="SignBridge Verified Demonstration",
                license="CC-BY / C-UDA Educational",
                notes="Fallback built-in sample",
            )
            self._entries.append(entry)
            self._id_index[entry.id] = entry

    @property
    def categories(self) -> list[str]:
        """Returns the list of available categories."""
        return list(self._categories)

    @property
    def total_count(self) -> int:
        """Returns total number of registered signs."""
        return len(self._entries)

    @property
    def available_video_count(self) -> int:
        """Returns number of signs with verified playable video."""
        return sum(1 for e in self._entries if e.available)

    def list_signs(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
        available_only: bool = False,
    ) -> list[CatalogSignEntry]:
        """Filters signs by category, text query, and video availability."""
        results = []
        target_cat = (category or "").strip()
        is_all_cats = not target_cat or target_cat.lower() in ("all", "all signs")

        for entry in self._entries:
            if not is_all_cats and entry.category.lower() != target_cat.lower():
                continue
            if available_only and not entry.available:
                continue
            if query and not entry.matches_query(query):
                continue
            results.append(entry)
        return results

    def paginate(
        self,
        signs: Sequence[CatalogSignEntry],
        page: int,
        page_size: int = 5,
    ) -> tuple[list[CatalogSignEntry], int, int]:
        """Paginates a sequence of signs. Returns (page_items, total_pages, clamped_page)."""
        if not signs:
            return [], 1, 1

        total = len(signs)
        total_pages = max(1, (total + page_size - 1) // page_size)
        clamped_page = max(1, min(page, total_pages))

        start_idx = (clamped_page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        return list(signs[start_idx:end_idx]), total_pages, clamped_page

    def get_sign_by_id(self, sign_id: str) -> Optional[CatalogSignEntry]:
        """Looks up a sign by its exact ID."""
        return self._id_index.get(sign_id)

    def get_sign_by_name(
        self,
        name: str,
        category: Optional[str] = None,
    ) -> Optional[CatalogSignEntry]:
        """Looks up a sign by its name or alias, optionally scoped to a category."""
        target_name = name.strip().upper()
        target_cat = category.strip().lower() if category else None

        # Clean target name of subtitle prefixes/suffixes
        clean_target = target_name.split("—")[0].split("-")[0].strip()

        for entry in self._entries:
            entry_upper = entry.name.upper()
            clean_entry = entry_upper.split("—")[0].split("-")[0].strip()

            is_match = (entry_upper == target_name or clean_entry == clean_target)
            if not is_match and ("/" in entry_upper or "/" in target_name or "—" in target_name):
                # Check alias components (e.g. 'GOODBYE' or 'BYE' in 'BYE / GOODBYE')
                components_entry = [c.strip() for c in clean_entry.split("/")]
                components_target = [c.strip() for c in clean_target.split("/")]
                if any(t in components_entry for t in components_target) or clean_target in components_entry:
                    is_match = True

            if is_match:
                if target_cat and target_cat not in ("all", "all signs"):
                    if entry.category.lower() == target_cat:
                        return entry
                else:
                    return entry
        return None

    def get_next_sign(
        self,
        current_id: str,
        category: Optional[str] = None,
        available_only: bool = True,
    ) -> Optional[CatalogSignEntry]:
        """Returns the next playable sign after current_id within the active category."""
        subset = self.list_signs(category=category, available_only=available_only)
        if not subset:
            return None
        idx = next((i for i, s in enumerate(subset) if s.id == current_id), -1)
        if idx == -1:
            return subset[0]
        next_idx = (idx + 1) % len(subset)
        return subset[next_idx]

    def get_prev_sign(
        self,
        current_id: str,
        category: Optional[str] = None,
        available_only: bool = True,
    ) -> Optional[CatalogSignEntry]:
        """Returns the previous playable sign before current_id within the active category."""
        subset = self.list_signs(category=category, available_only=available_only)
        if not subset:
            return None
        idx = next((i for i, s in enumerate(subset) if s.id == current_id), -1)
        if idx == -1:
            return subset[-1]
        prev_idx = (idx - 1) % len(subset)
        return subset[prev_idx]
