"""Comprehensive test suite for Sign Language Reference Library and Demo Hub."""

from pathlib import Path
import numpy as np
import pytest

from signbridge.ui.app import SignBridgeApp
from signbridge.ui.layout import DashboardLayout
from signbridge.ui.overlays import ModalOverlayManager
from signbridge.ui.state import ActiveModal, UIState
from signbridge.ui.theme import OBSIDIAN_THEME, UITheme
from signbridge.video.catalog import CatalogSignEntry, SignLanguageCatalog
from signbridge.video.sample_manager import SampleVideoManager


@pytest.fixture
def catalog():
    """Returns an initialized SignLanguageCatalog instance."""
    return SignLanguageCatalog()


# =============================================================================
# 1. Catalog Loading & Schema Integrity Tests
# =============================================================================

def test_catalog_loads_successfully(catalog):
    """Verifies the catalog loads real entries and has expected categories."""
    assert catalog.total_count >= 75
    assert catalog.available_video_count >= 30
    assert "All Signs" in catalog.categories
    assert "Alphabet" in catalog.categories
    assert "Numbers" in catalog.categories
    assert "Greetings" in catalog.categories
    assert "Common Words" in catalog.categories
    assert "Daily Communication" in catalog.categories
    assert "Common Phrases" in catalog.categories


def test_catalog_alphabet_integrity(catalog):
    """Verifies that all 26 letters A-Z are present, have valid descriptions, and valid video files."""
    alpha_signs = catalog.list_signs(category="Alphabet")
    assert len(alpha_signs) == 26
    letters = [s.name for s in alpha_signs]
    expected_letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    assert letters == expected_letters

    for s in alpha_signs:
        assert s.type == "letter"
        assert s.language == "ASL"
        assert len(s.description) > 10
        assert s.available is True
        assert s.video is not None
        vpath = s.get_full_video_path()
        assert vpath is not None
        assert vpath.exists()
        assert vpath.is_file()


def test_catalog_numbers_integrity(catalog):
    """Verifies numbers 0-9 are present with authentic descriptions and pending video flag."""
    num_signs = catalog.list_signs(category="Numbers")
    assert len(num_signs) == 10
    nums = [s.name for s in num_signs]
    assert nums == [str(i) for i in range(10)]

    for s in num_signs:
        assert s.type == "number"
        assert len(s.description) > 10
        assert s.available is False  # Video pending curation


def test_catalog_greetings_integrity(catalog):
    """Verifies greetings category contains hello, thank you, goodbye, etc."""
    greetings = catalog.list_signs(category="Greetings")
    names = [s.name for s in greetings]
    assert "Hello" in names
    assert "Thank You" in names
    assert any("Goodbye" in n for n in names)

    # Hello and Thank you should have verified demonstration videos
    hello_entry = catalog.get_sign_by_name("Hello", category="Greetings")
    assert hello_entry is not None
    assert hello_entry.available is True
    assert hello_entry.get_full_video_path() is not None

    # Goodbye has authentic educational guide and verified demonstration video
    goodbye_entry = catalog.get_sign_by_name("Goodbye", category="Greetings")
    assert goodbye_entry is not None
    assert "Bye / Goodbye" in goodbye_entry.name
    assert "shoulder" in goodbye_entry.description.lower() or "wave" in goodbye_entry.description.lower()
    assert goodbye_entry.available is True
    assert goodbye_entry.get_full_video_path() is not None


def test_catalog_common_words_integrity(catalog):
    """Verifies common words category contains essential vocabulary and verified videos."""
    words = catalog.list_signs(category="Common Words")
    names = [s.name for s in words]
    for required in ["Help", "Yes", "No", "Stop", "Please", "Sorry", "Water", "Food"]:
        assert required in names

    # Verified videos exist for all core vocabulary signs
    for vname in ["Help", "Yes", "No", "Stop", "Please", "Sorry", "Water", "Food"]:
        entry = catalog.get_sign_by_name(vname, category="Common Words")
        assert entry is not None
        assert entry.available is True
        assert entry.get_full_video_path() is not None


def test_catalog_fallback_handling(tmp_path):
    """Verifies catalog degrades gracefully when catalog file is missing."""
    empty_catalog = SignLanguageCatalog(catalog_path=tmp_path / "non_existent.json")
    assert empty_catalog.total_count >= 6
    assert any(s.name == "HELLO" for s in empty_catalog.list_signs())


# =============================================================================
# 2. Search & Filter Tests
# =============================================================================

def test_catalog_category_filtering(catalog):
    """Verifies category filtering restricts returned signs to only that category."""
    alpha = catalog.list_signs(category="Alphabet")
    assert all(s.category == "Alphabet" for s in alpha)

    comm = catalog.list_signs(category="Daily Communication")
    assert all(s.category == "Daily Communication" for s in comm)
    assert len(comm) >= 10


def test_catalog_text_query_search(catalog):
    """Verifies case-insensitive substring search matching name and description."""
    results = catalog.list_signs(query="salute")
    assert any(s.name == "Hello" for s in results)

    results_a = catalog.list_signs(query="pirate hook")
    assert any(s.name == "X" for s in results_a)


def test_catalog_available_only_filter(catalog):
    """Verifies filtering by available_only returns only signs with playable videos."""
    avail = catalog.list_signs(available_only=True)
    assert len(avail) >= 30
    assert all(s.available is True for s in avail)
    assert all(s.get_full_video_path() is not None for s in avail)


# =============================================================================
# 3. Pagination Tests
# =============================================================================

def test_catalog_pagination(catalog):
    """Verifies pagination slicing and page clamping."""
    all_signs = catalog.list_signs()
    page_1, total_pages, cur_1 = catalog.paginate(all_signs, page=1, page_size=5)
    assert len(page_1) == 5
    assert cur_1 == 1
    assert total_pages >= 15

    # Out of bounds page clamps to total_pages
    page_last, _, cur_last = catalog.paginate(all_signs, page=999, page_size=5)
    assert cur_last == total_pages
    assert len(page_last) > 0


# =============================================================================
# 4. Navigation Tests
# =============================================================================

def test_catalog_next_prev_navigation(catalog):
    """Verifies cyclic next and previous navigation among playable signs."""
    entry_a = catalog.get_sign_by_name("A")
    assert entry_a is not None

    next_sign = catalog.get_next_sign(entry_a.id, available_only=True)
    assert next_sign is not None
    assert next_sign.name == "B"

    prev_sign = catalog.get_prev_sign(next_sign.id, available_only=True)
    assert prev_sign is not None
    assert prev_sign.name == "A"


# =============================================================================
# 5. SampleVideoManager Integration Tests
# =============================================================================

def test_sample_manager_catalog_integration():
    """Verifies SampleVideoManager looks up signs across extended catalog categories."""
    mgr = SampleVideoManager()
    # Built-in word sign
    rec_hello = mgr.get_sample_by_sign("HELLO")
    assert rec_hello is not None
    assert rec_hello.sign == "HELLO"

    # Alphabet sign resolved via catalog
    rec_a = mgr.get_sample_by_sign("A")
    assert rec_a is not None
    assert rec_a.sign == "A"
    assert "alphabet" in rec_a.full_path.lower()
    assert rec_a.duration_seconds > 1.0

    # ID lookup
    rec_c = mgr.get_sample_by_id("asl_alpha_c")
    assert rec_c is not None
    assert rec_c.sign == "C"


# =============================================================================
# 6. UI & Overlay Rendering Tests
# =============================================================================

def test_overlay_renders_demo_hub_modal():
    """Verifies ModalOverlayManager renders the enhanced Demo Hub with categories and buttons."""
    manager = ModalOverlayManager()
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    ui_state = UIState()
    ui_state.open_modal(ActiveModal.VIDEO_DEMO)
    theme = OBSIDIAN_THEME

    manager.render(canvas, ui_state, theme, {})
    buttons = manager.get_registered_buttons()

    # Category tab buttons registered
    assert "btn_cat_all" in buttons
    assert "btn_cat_alpha" in buttons
    assert "btn_cat_nums" in buttons
    assert "btn_cat_greetings" in buttons
    assert "btn_cat_common" in buttons
    assert "btn_cat_daily" in buttons
    assert "btn_cat_phrases" in buttons

    # Modal controls registered
    assert "modal_close" in buttons
    assert "modal_close_btn" in buttons
    assert "btn_source_camera" in buttons
    assert "btn_page_prev" in buttons
    assert "btn_page_next" in buttons

    # Sign cards registered
    play_buttons = [b for b in buttons if b.startswith("btn_play_sign_")]
    assert len(play_buttons) >= 5


def test_overlay_renders_detail_view():
    """Verifies ModalOverlayManager renders the detailed reference view."""
    manager = ModalOverlayManager()
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    ui_state = UIState()
    ui_state.open_modal(ActiveModal.VIDEO_DEMO)
    ui_state.demo_selected_sign_id = "asl_word_help"
    ui_state.demo_view_mode = "DETAIL"
    theme = OBSIDIAN_THEME

    manager.render(canvas, ui_state, theme, {})
    buttons = manager.get_registered_buttons()

    assert "btn_demo_back_list" in buttons
    assert "btn_play_sign_asl_word_help" in buttons


def test_layout_video_transport_navigation_buttons():
    """Verifies layout renders next and previous demonstration video buttons in transport bar."""
    layout = DashboardLayout()
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    ui_state = UIState()
    ui_state.active_input_source = "DEMO"
    theme = OBSIDIAN_THEME

    from signbridge.pipeline.state import PipelineRuntimeState
    p_state = PipelineRuntimeState(frame_source="DEMO_VIDEO")

    layout._render_camera_panel(canvas, 10, 10, 800, 600, canvas[0:480, 0:640], p_state, ui_state, theme)
    buttons = layout.get_registered_buttons()

    assert "btn_vid_prev_sample" in buttons
    assert "btn_vid_next_sample" in buttons
    assert "btn_vid_play" in buttons
    assert "btn_vid_pause" in buttons


# =============================================================================
# 7. App Controller & Event Dispatching Tests
# =============================================================================

def test_app_demo_hub_category_switching():
    """Verifies category selection buttons update ui_state category and reset page."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)

    app._dispatch_button_action("btn_cat_alpha")
    assert app.ui_state.demo_category == "Alphabet"
    assert app.ui_state.demo_page == 1

    app._dispatch_button_action("btn_cat_nums")
    assert app.ui_state.demo_category == "Numbers"
    assert app.ui_state.demo_page == 1

    app._dispatch_button_action("btn_cat_all")
    assert app.ui_state.demo_category == "All Signs"


def test_app_demo_hub_pagination_and_filter_reset():
    """Verifies next/prev pagination buttons and search reset."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)

    app.ui_state.demo_page = 1
    app._dispatch_button_action("btn_page_next")
    assert app.ui_state.demo_page == 2

    app._dispatch_button_action("btn_page_prev")
    assert app.ui_state.demo_page == 1

    # Reset filter
    app.ui_state.demo_search_query = "wave"
    app.ui_state.demo_category = "Greetings"
    app._dispatch_button_action("btn_search_clear")
    assert app.ui_state.demo_search_query == ""
    assert app.ui_state.demo_category == "All Signs"


def test_app_demo_hub_detail_view_navigation():
    """Verifies opening sign details and returning back to the list view."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)

    app._dispatch_button_action("btn_detail_sign_asl_word_help")
    assert app.ui_state.demo_selected_sign_id == "asl_word_help"
    assert app.ui_state.demo_view_mode == "DETAIL"

    app._dispatch_button_action("btn_demo_back_list")
    assert app.ui_state.demo_view_mode == "LIST"


def test_app_demo_hub_play_available_sign():
    """Verifies clicking play on an available sign switches pipeline to demo video."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)
    app.ui_state.open_modal(ActiveModal.VIDEO_DEMO)

    app._dispatch_button_action("btn_play_sign_asl_word_hello")
    mock_pipeline.switch_to_demo_video.assert_called_once()
    assert app.ui_state.active_input_source == "DEMO"
    assert app.ui_state.active_modal == ActiveModal.NONE
    assert "HELLO" in app.ui_state.selected_sample_video.upper()


def test_app_demo_hub_missing_video_handling():
    """Verifies clicking a sign without video opens detail reference and shows educational toast."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)
    app.ui_state.open_modal(ActiveModal.VIDEO_DEMO)

    # Number 1 has no recorded video
    app._dispatch_button_action("btn_play_sign_asl_num_1")
    # Pipeline should NOT switch source
    mock_pipeline.switch_to_demo_video.assert_not_called()
    # Opens detail view for reference
    assert app.ui_state.demo_selected_sign_id == "asl_num_1"
    assert app.ui_state.demo_view_mode == "DETAIL"
    assert app.ui_state.toast is not None
    assert "Demo video unavailable" in app.ui_state.toast.message


def test_app_video_transport_sample_cycling():
    """Verifies transport bar next/prev advances through playable demonstration videos."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)
    app.ui_state.active_input_source = "DEMO"
    app.ui_state.selected_sample_video = "A"

    app._dispatch_button_action("btn_vid_next_sample")
    mock_pipeline.switch_to_demo_video.assert_called()
    assert app.ui_state.selected_sample_video == "B"


def test_asl_goodbye_educational_entry_and_aliases(catalog):
    """Verifies Goodbye is clearly labeled 'Bye / Goodbye — American Sign Language (ASL)' and resolves via aliases."""
    # Lookup by full official name
    entry_full = catalog.get_sign_by_name("Bye / Goodbye — American Sign Language (ASL)")
    assert entry_full is not None
    assert entry_full.id == "asl_greet_goodbye"
    assert entry_full.category == "Greetings"
    assert entry_full.language == "ASL"

    # Lookup by short aliases
    entry_bye = catalog.get_sign_by_name("Bye")
    entry_goodbye = catalog.get_sign_by_name("Goodbye")
    assert entry_bye is not None
    assert entry_goodbye is not None
    assert entry_bye.id == "asl_greet_goodbye"
    assert entry_goodbye.id == "asl_greet_goodbye"

    # Mechanics specification describes shoulder or head height wave (NOT stomach)
    desc = entry_goodbye.description.lower()
    assert "shoulder" in desc or "head" in desc
    assert "wave" in desc or "open palm" in desc

    # Educational notes document attribution and licensed MP4 guidance
    assert "aisa.solutions" in (entry_goodbye.notes or "").lower()
    assert "lifeprint" in (entry_goodbye.notes or "").lower()


def test_category_isolation_in_transport_navigation():
    """Verifies next/prev navigation in transport bar stays strictly within the active category."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)
    app.ui_state.active_input_source = "DEMO"

    # When playing HELLO in Greetings, next must cycle to THANK YOU (in Greetings)
    app.ui_state.selected_sample_video = "HELLO"
    app._dispatch_button_action("btn_vid_next_sample")
    assert app.ui_state.selected_sample_video == "THANK YOU"
    assert app.ui_state.demo_category == "Greetings"

    # Cycle next again: should advance to Goodbye (verified video in Greetings)
    app._dispatch_button_action("btn_vid_next_sample")
    assert "GOODBYE" in app.ui_state.selected_sample_video.upper()
    assert app.ui_state.demo_category == "Greetings"

    # Cycle next again: wraps back to HELLO in Greetings
    app._dispatch_button_action("btn_vid_next_sample")
    assert app.ui_state.selected_sample_video == "HELLO"
    assert app.ui_state.demo_category == "Greetings"

    # When playing Alphabet A, next must cycle to B, NOT Greetings or Common Words
    app.ui_state.selected_sample_video = "A"
    app._dispatch_button_action("btn_vid_next_sample")
    assert app.ui_state.selected_sample_video == "B"
    assert app.ui_state.demo_category == "Alphabet"


def test_missing_video_card_next_button_same_category():
    """Verifies cards for signs without videos provide Next Sign button that stays within the same category."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)
    app.ui_state.open_modal(ActiveModal.VIDEO_DEMO)
    app.ui_state.demo_category = "Numbers"
    app.ui_state.demo_page = 1

    # In Numbers, all signs are reference guides without videos.
    # Clicking btn_next_card_asl_num_0 should advance to Number 1 within Numbers
    app._dispatch_button_action("btn_next_card_asl_num_0")
    assert app.ui_state.demo_category == "Numbers"
    assert app.ui_state.toast is not None
    assert "Next sign: '1'" in app.ui_state.toast.message


def test_missing_video_detail_navigation_same_category():
    """Verifies detail view next/prev buttons cycle within the same category."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    app = SignBridgeApp(pipeline=mock_pipeline)
    app.ui_state.open_modal(ActiveModal.VIDEO_DEMO)
    app.ui_state.demo_category = "Greetings"
    app.ui_state.demo_selected_sign_id = "asl_greet_goodbye"
    app.ui_state.demo_view_mode = "DETAIL"

    # In Greetings, next from Goodbye advances to next sign in Greetings
    app._dispatch_button_action("btn_demo_next_sign")
    assert app.ui_state.demo_view_mode == "DETAIL"
    assert app.ui_state.demo_selected_sign_id == "asl_greet_good_morning"
    assert app.ui_state.toast is not None
    assert "Good Morning" in app.ui_state.toast.message

    # Previous returns back to Goodbye
    app._dispatch_button_action("btn_demo_prev_sign")
    assert app.ui_state.demo_selected_sign_id == "asl_greet_goodbye"


def test_category_with_zero_videos_renders_without_crashing():
    """Verifies that an entire category with 0 videos (e.g. Numbers) renders safely and registers buttons."""
    manager = ModalOverlayManager()
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    ui_state = UIState()
    ui_state.open_modal(ActiveModal.VIDEO_DEMO)
    ui_state.demo_category = "Numbers"
    theme = OBSIDIAN_THEME

    manager.render(canvas, ui_state, theme, {})
    buttons = manager.get_registered_buttons()

    # Verify buttons for cards registered
    next_card_btns = [b for b in buttons if b.startswith("btn_next_card_")]
    assert len(next_card_btns) >= 5
    guide_btns = [b for b in buttons if b.startswith("btn_detail_sign_")]
    assert len(guide_btns) >= 5


