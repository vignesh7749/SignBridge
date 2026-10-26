"""Management of built-in demonstration samples and user-provided video files."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
import time
from typing import Optional, Sequence, Union

from signbridge.core.logger import get_logger
from signbridge.video.base import VideoMetadata, VideoValidationResult
from signbridge.video.validator import validate_video_file

logger = get_logger("video.sample_manager")


@dataclass(frozen=True)
class SampleVideoRecord:
    """Metadata record for a registered demonstration or user video sample."""
    file: str
    sign: str
    source: str
    license: str
    source_url: str
    duration_seconds: float
    verified: bool
    is_user_sample: bool = False
    full_path: Optional[str] = None
    width: int = 0
    height: int = 0
    fps: float = 30.0
    frame_count: int = 0

    @property
    def filename(self) -> str:
        """Alias for file attribute."""
        return self.file

    @property
    def gloss(self) -> str:
        """Alias for sign attribute."""
        return self.sign

    @property
    def duration(self) -> float:
        """Alias for duration_seconds."""
        return self.duration_seconds

    @property
    def verification_status(self) -> str:
        """String representation of verification status."""
        return "VERIFIED" if self.verified else "UNVERIFIED"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["filename"] = self.filename
        data["gloss"] = self.gloss
        data["duration"] = self.duration
        data["verification_status"] = self.verification_status
        return data


class SampleVideoManager:
    """Central registry and controller for built-in and user-uploaded sample videos."""

    def __init__(
        self,
        samples_dir: Optional[Union[str, Path]] = None,
        user_samples_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.samples_dir = Path(samples_dir or (project_root / "datasets" / "samples" / "word_signs")).resolve()
        self.user_samples_dir = Path(user_samples_dir or (project_root / "datasets" / "user_samples" / "word_signs")).resolve()

        self.manifest_file = self.samples_dir / "manifest.json"
        self.samples_metadata_file = self.samples_dir / "samples_metadata.json"
        self.user_catalog_file = self.user_samples_dir / "user_catalog.json"

        # Ensure directories exist
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.user_samples_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Built-in Samples Management
    # -------------------------------------------------------------------------

    def list_builtin_samples(self) -> list[SampleVideoRecord]:
        """Loads and returns all validated built-in demonstration samples."""
        records: list[SampleVideoRecord] = []

        # Check manifest.json first, then samples_metadata_file
        target_manifest = self.manifest_file if self.manifest_file.exists() else self.samples_metadata_file
        if not target_manifest.exists():
            logger.debug("Built-in sample manifest file does not exist: %s", target_manifest)
            return records

        try:
            with open(target_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data.get("samples", []):
                rel_file = entry.get("filename") or entry.get("file", "")
                target_path = self.samples_dir / rel_file

                # Validate physical file presence and integrity
                val = validate_video_file(target_path)
                if val.is_valid and val.metadata is not None:
                    sign_name = (entry.get("sign") or entry.get("gloss") or "UNKNOWN").upper()
                    is_ver = entry.get("verification_status") == "VERIFIED" or bool(entry.get("verified", True))
                    records.append(
                        SampleVideoRecord(
                            file=rel_file,
                            sign=sign_name,
                            source=entry.get("source", "SignBridge Verified Demonstration"),
                            license=entry.get("license", "CC-BY / C-UDA Educational"),
                            source_url=entry.get("source_url", ""),
                            duration_seconds=val.metadata.duration_seconds,
                            verified=entry.get("verified", True),
                            is_user_sample=False,
                            full_path=str(target_path),
                            width=val.metadata.width,
                            height=val.metadata.height,
                            fps=val.metadata.fps,
                            frame_count=val.metadata.frame_count,
                        )
                    )
                else:
                    logger.debug("Built-in sample invalid or missing: %s (%s)", rel_file, val.error_message)

        except Exception as e:
            logger.error("Error reading built-in samples metadata: %s", e)

        return records

    def get_sample_by_sign(self, sign_name: str) -> Optional[SampleVideoRecord]:
        """Finds a built-in demonstration sample matching the given sign name."""
        sign_upper = sign_name.strip().upper()
        for sample in self.list_builtin_samples():
            if sample.sign == sign_upper:
                return sample
        return None

    # -------------------------------------------------------------------------
    # User-Uploaded Samples Management
    # -------------------------------------------------------------------------

    def list_user_samples(self) -> list[SampleVideoRecord]:
        """Loads and returns all registered and validated user-uploaded test videos."""
        records: list[SampleVideoRecord] = []

        if not self.user_catalog_file.exists():
            return records

        try:
            with open(self.user_catalog_file, "r", encoding="utf-8") as f:
                catalog = json.load(f)

            for item in catalog.get("user_videos", []):
                filename = item.get("file", "")
                target_path = self.user_samples_dir / filename

                val = validate_video_file(target_path)
                if val.is_valid and val.metadata is not None:
                    records.append(
                        SampleVideoRecord(
                            file=filename,
                            sign=item.get("sign", "CUSTOM").upper(),
                            source=item.get("source", "User Uploaded Sample"),
                            license="User Provided",
                            source_url=item.get("original_path", ""),
                            duration_seconds=val.metadata.duration_seconds,
                            verified=True,
                            is_user_sample=True,
                            full_path=str(target_path),
                            width=val.metadata.width,
                            height=val.metadata.height,
                            fps=val.metadata.fps,
                            frame_count=val.metadata.frame_count,
                        )
                    )
        except Exception as e:
            logger.error("Error reading user sample catalog: %s", e)

        return records

    def add_user_sample(
        self,
        source_video_path: Union[str, Path],
        sign_label: str = "CUSTOM",
    ) -> VideoValidationResult:
        """Validates and imports a local user video into datasets/user_samples/word_signs/.

        Strict Boundary: User sample videos are segregated from the official
        training dataset and stored exclusively for demo and testing inference.
        """
        source_path = Path(source_video_path).resolve()
        val = validate_video_file(source_path)
        if not val.is_valid or val.metadata is None:
            return val

        try:
            timestamp_prefix = int(time.time())
            safe_name = f"{timestamp_prefix}_{source_path.name}"
            dest_path = self.user_samples_dir / safe_name
            shutil.copy2(source_path, dest_path)

            # Update user catalog
            catalog = {"version": "1.0", "user_videos": []}
            if self.user_catalog_file.exists():
                try:
                    with open(self.user_catalog_file, "r", encoding="utf-8") as f:
                        catalog = json.load(f)
                except Exception:
                    pass

            new_entry = {
                "file": safe_name,
                "sign": sign_label.strip().upper(),
                "original_path": str(source_path),
                "added_timestamp": time.time(),
                "duration_seconds": val.metadata.duration_seconds,
            }
            catalog.setdefault("user_videos", []).append(new_entry)

            with open(self.user_catalog_file, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2)

            logger.info("User sample added successfully: %s -> %s", source_path.name, safe_name)
            return val

        except Exception as e:
            msg = f"Failed to import user video: {e}"
            logger.error(msg)
            return VideoValidationResult(is_valid=False, error_message=msg)

    def remove_user_sample(self, filename: str) -> bool:
        """Removes a user video from datasets/user_samples/word_signs/ and updates catalog."""
        target_path = self.user_samples_dir / filename
        removed = False

        if target_path.exists():
            try:
                target_path.unlink()
                removed = True
            except Exception as e:
                logger.warning("Could not delete user sample file %s: %e", filename, e)

        if self.user_catalog_file.exists():
            try:
                with open(self.user_catalog_file, "r", encoding="utf-8") as f:
                    catalog = json.load(f)

                videos = catalog.get("user_videos", [])
                catalog["user_videos"] = [v for v in videos if v.get("file") != filename]

                with open(self.user_catalog_file, "w", encoding="utf-8") as f:
                    json.dump(catalog, f, indent=2)
                removed = True
            except Exception as e:
                logger.error("Error updating user catalog upon removal: %s", e)

        return removed
