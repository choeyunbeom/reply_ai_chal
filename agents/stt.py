"""
agents/stt.py
-------------
Role C — Speech-to-Text Agent.

Preprocesses audio files into text transcripts at level start. Runs eagerly
(all files transcribed upfront), caches results, and exposes a lookup
interface that Context consumes during bundle assembly.

Zero LLM budget impact: uses local Whisper model (free compute).

Filename convention (from L3 data):
    YYYYMMDD_HHMMSS-firstname_lastname.mp3
    e.g. 20870117_010505-guido_do_êhn.mp3
    → date: 2087-01-17, time: 01:05:05, speaker: guido do êhn

Public interface:
    stt.transcribe_all(audio_dir)     Orchestrator, at level start
    stt.get_transcript(user_id)       Context, during bundle assembly
    stt.get_transcript_by_name(name)  Context, fallback by speaker name
    stt.get_all_transcripts()         Investigator, for bulk access
    stt.fraud_signals(user_id)        Context, for bundle field

Per CLAUDE.md:
    - logging not print
    - pathlib.Path for all file access
    - No LLM calls
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fraud keyword lists for voice content
# ---------------------------------------------------------------------------

# Reuse the text-based phishing fragments from Context for consistency
KNOWN_PHISHING_FRAGMENTS = [
    "paypa1", "amaz0n", "netfl1x", "ub3r", "micros0ft", "app1e", "g00gle",
    "paypa1-secure", "amaz0n-verify", "netfl1x-bill", "ub3r-verify",
]

# Standard fraud keywords (shared with SMS/email scanning)
FRAUD_KEYWORDS = [
    "verify now", "verify your account", "verify identity",
    "suspicious login", "suspicious sign-in", "unusual login",
    "account will be locked", "account suspended", "prevent lock",
    "update payment", "billing information", "payment details",
    "urgent", "immediately", "act now",
]

# Voice-specific fraud keywords: social engineering scripts spoken over phone
VOICE_FRAUD_KEYWORDS = [
    "this is your bank",
    "calling from the bank",
    "calling from your bank",
    "fraud department",
    "security department",
    "security team",
    "confirm your identity",
    "confirm your pin",
    "confirm your password",
    "one-time code",
    "one time code",
    "verification code",
    "give me the code",
    "read me the code",
    "transfer the funds",
    "move the money",
    "safe account",
    "holding account",
    "your account has been compromised",
    "your account is at risk",
    "unauthorized transaction",
    "unauthorised transaction",
    "we need to verify",
    "for your protection",
    "gift card",
    "itunes card",
    "google play card",
]

ALL_VOICE_KEYWORDS = FRAUD_KEYWORDS + VOICE_FRAUD_KEYWORDS


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_audio_filename(filepath: Path) -> dict[str, Any]:
    """
    Extract metadata from the audio filename convention.

    Expected: YYYYMMDD_HHMMSS-firstname_lastname.mp3
    Returns: {date, time, datetime, speaker_name, speaker_parts}
    """
    stem = filepath.stem  # e.g. "20870117_010505-guido_do_êhn"

    # Split on first hyphen that separates timestamp from name
    # Handle names with hyphens by splitting on the timestamp-name boundary
    match = re.match(r"^(\d{8})_(\d{6})-(.+)$", stem)
    if not match:
        logger.warning(f"Cannot parse audio filename: {filepath.name}")
        return {
            "date": None, "time": None, "datetime": None,
            "speaker_name": stem, "speaker_parts": [],
        }

    date_str, time_str, name_raw = match.groups()

    try:
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    except ValueError:
        dt = None

    # Convert underscores to spaces for the speaker name
    speaker_name = name_raw.replace("_", " ")

    return {
        "date": date_str,
        "time": time_str,
        "datetime": dt,
        "speaker_name": speaker_name,
        "speaker_parts": speaker_name.lower().split(),
    }


# ---------------------------------------------------------------------------
# Transcript data structure
# ---------------------------------------------------------------------------

@dataclass
class Transcript:
    """A single transcribed audio file with metadata."""
    filepath: str
    speaker_name: str
    timestamp: datetime | None
    text: str
    language: str = "unknown"
    duration_seconds: float = 0.0
    fraud_signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "speaker_name": self.speaker_name,
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "text": self.text,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "fraud_signals": self.fraud_signals,
        }


# ---------------------------------------------------------------------------
# Whisper backend
# ---------------------------------------------------------------------------

class WhisperBackend:
    """
    Local Whisper STT backend. Zero API cost.

    Tries faster-whisper first (faster on CPU), falls back to openai-whisper,
    falls back to a stub that returns empty transcripts.

    Model size selection:
        'tiny'  — 75MB, ~10x realtime on CPU, sufficient for keyword detection
        'base'  — 140MB, ~7x realtime, better accuracy if tiny is too noisy
        'small' — 460MB, needs more RAM, only if base fails on the audio quality
    """

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None
        self._backend_type: str | None = None
        self._init_backend()

    def _init_backend(self):
        """Try available backends in order of preference."""
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
            self._backend_type = "faster-whisper"
            logger.info(f"STT: using faster-whisper ({self.model_size})")
            return
        except Exception as e:
            logger.debug(f"faster-whisper unavailable: {e}")

        # Try openai-whisper
        try:
            import whisper
            self._model = whisper.load_model(self.model_size)
            self._backend_type = "openai-whisper"
            logger.info(f"STT: using openai-whisper ({self.model_size})")
            return
        except Exception as e:
            logger.debug(f"openai-whisper unavailable: {e}")

        # No backend available
        self._backend_type = None
        logger.warning(
            "STT: no Whisper backend available. "
            "Transcripts will be empty. Install faster-whisper or openai-whisper."
        )

    def transcribe(self, filepath: Path) -> dict[str, Any]:
        """
        Transcribe a single audio file.

        Returns: {text: str, language: str, duration: float}
        """
        if self._backend_type is None:
            duration = self._get_duration(filepath)
            return {"text": "", "language": "unknown", "duration": duration}

        if self._backend_type == "faster-whisper":
            return self._transcribe_faster(filepath)
        else:
            return self._transcribe_openai(filepath)

    def _transcribe_faster(self, filepath: Path) -> dict[str, Any]:
        segments, info = self._model.transcribe(str(filepath))
        text = " ".join(seg.text.strip() for seg in segments)
        return {
            "text": text,
            "language": info.language,
            "duration": info.duration,
        }

    def _transcribe_openai(self, filepath: Path) -> dict[str, Any]:
        result = self._model.transcribe(str(filepath))
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "duration": self._get_duration(filepath),
        }

    @staticmethod
    def _get_duration(filepath: Path) -> float:
        """Get audio duration via ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 str(filepath)],
                capture_output=True, text=True, timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Fraud signal scanning (mirrors Context's SMS/email scanning)
# ---------------------------------------------------------------------------

def scan_transcript_for_fraud(text: str) -> dict:
    """
    Scan a transcript for fraud signals.

    Same structure as Context's sms_fraud_signals / email_fraud_signals
    so it drops into the bundle seamlessly.
    """
    if not text:
        return {
            "phishing_hits": 0,
            "suspicious_domains": [],
            "fraud_keywords": [],
            "voice_fraud_keywords": [],
        }

    text_lower = text.lower()

    phishing_hits = 0
    suspicious_domains: list[str] = []
    fraud_kw_found: list[str] = []
    voice_kw_found: list[str] = []

    # Phishing domains (unlikely in voice but check transcription artefacts)
    for frag in KNOWN_PHISHING_FRAGMENTS:
        if frag in text_lower:
            phishing_hits += 1
            suspicious_domains.append(frag)

    # Standard fraud keywords
    for kw in FRAUD_KEYWORDS:
        if kw in text_lower:
            fraud_kw_found.append(kw)

    # Voice-specific fraud keywords
    for kw in VOICE_FRAUD_KEYWORDS:
        if kw in text_lower:
            voice_kw_found.append(kw)
            # Voice fraud keywords also count as phishing hits
            # since they indicate social engineering
            phishing_hits += 1

    return {
        "phishing_hits": phishing_hits,
        "suspicious_domains": list(set(suspicious_domains)),
        "fraud_keywords": list(set(fraud_kw_found)),
        "voice_fraud_keywords": list(set(voice_kw_found)),
    }


# ---------------------------------------------------------------------------
# User matching
# ---------------------------------------------------------------------------

def match_speaker_to_user(
    speaker_name: str,
    users: list[dict],
    sender_ids: list[str] | None = None,
) -> str | None:
    """
    Match an audio speaker name to a user ID.

    Strategy (in order):
      1. Exact first_name + last_name match against users.json
      2. First-name-only match (handles partial names in filenames)
      3. Fuzzy substring match on sender_id components

    Returns the user's sender_id if matched, None otherwise.
    """
    speaker_lower = speaker_name.lower().strip()
    speaker_parts = set(speaker_lower.split())

    # Strategy 1: exact name match against users.json
    for user in users:
        first = user.get("first_name", "").lower()
        last = user.get("last_name", "").lower()
        full = f"{first} {last}"
        if speaker_lower == full:
            # Need to map user to sender_id via IBAN
            return _user_to_sender_id(user, sender_ids)

    # Strategy 2: first-name match
    for user in users:
        first = user.get("first_name", "").lower()
        if first and first in speaker_parts:
            return _user_to_sender_id(user, sender_ids)

    # Strategy 3: check if speaker name parts appear in sender_ids
    if sender_ids:
        for sid in sender_ids:
            sid_parts = set(sid.lower().replace("-", " ").split())
            if speaker_parts & sid_parts:
                return sid

    return None


def _user_to_sender_id(user: dict, sender_ids: list[str] | None) -> str | None:
    """Map a users.json entry to its sender_id in the transaction data."""
    iban = user.get("iban", "")
    if not sender_ids:
        return None

    # The sender_id encoding often includes name fragments
    # e.g. RGNR-LNAA-7FF-AUD-0 for Alain Regnier in Audincourt
    first = user.get("first_name", "").lower()
    last = user.get("last_name", "").lower()
    city = user.get("residence", {}).get("city", "").lower()

    for sid in sender_ids:
        sid_lower = sid.lower()
        # Check if name or city fragments appear in the sender_id
        if (first[:3] in sid_lower or last[:3] in sid_lower
                or city[:3] in sid_lower):
            return sid

    return None


# ---------------------------------------------------------------------------
# STT Agent facade
# ---------------------------------------------------------------------------

class STTAgent:
    """
    Speech-to-Text Agent. Transcribes audio files at level start and
    exposes transcripts for Context and Investigator consumption.

    Usage:
        stt = STTAgent(users=users_list, sender_ids=sender_id_list)
        stt.transcribe_all(Path("data/level_3/audio"))
        transcript = stt.get_transcript_by_name("gabriel chauvin")
        signals = stt.fraud_signals_by_name("gabriel chauvin")

    Integration with Context:
        Context calls stt.fraud_signals(user_id) and adds the result
        as an "audio_fraud_signals" field in the evidence bundle.
        Context calls stt.get_transcript(user_id) to expose raw text
        to the Investigator for deeper reasoning.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        users: list[dict] | None = None,
        sender_ids: list[str] | None = None,
    ):
        self._backend = WhisperBackend(model_size=model_size)
        self._users = users or []
        self._sender_ids = sender_ids or []

        # Transcript storage: indexed by speaker name and user_id
        self._by_name: dict[str, list[Transcript]] = {}
        self._by_user_id: dict[str, list[Transcript]] = {}
        self._all_transcripts: list[Transcript] = []

    def transcribe_all(self, audio_dir: str | Path) -> int:
        """
        Transcribe all audio files in a directory. Call at level start.

        Returns the number of files successfully transcribed.
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            logger.warning(f"STT: audio directory not found: {audio_dir}")
            return 0

        audio_files = sorted(
            p for p in audio_dir.iterdir()
            if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg", ".flac")
        )

        if not audio_files:
            logger.info(f"STT: no audio files found in {audio_dir}")
            return 0

        logger.info(f"STT: transcribing {len(audio_files)} files from {audio_dir}")
        count = 0

        for filepath in audio_files:
            try:
                transcript = self._transcribe_file(filepath)
                if transcript:
                    self._index_transcript(transcript)
                    count += 1
            except Exception as e:
                logger.warning(f"STT: failed to transcribe {filepath.name}: {e}")

        logger.info(
            f"STT: transcribed {count}/{len(audio_files)} files, "
            f"{len(self._by_user_id)} matched to users"
        )
        return count

    def get_transcript(self, user_id: str) -> list[dict] | None:
        """Get transcripts for a user by their sender_id."""
        transcripts = self._by_user_id.get(user_id)
        if not transcripts:
            return None
        return [t.to_dict() for t in transcripts]

    def get_transcript_by_name(self, name: str) -> list[dict] | None:
        """Get transcripts by speaker name (case-insensitive)."""
        transcripts = self._by_name.get(name.lower().strip())
        if not transcripts:
            return None
        return [t.to_dict() for t in transcripts]

    def get_all_transcripts(self) -> list[dict]:
        """All transcripts. Used by Investigator for bulk access."""
        return [t.to_dict() for t in self._all_transcripts]

    def fraud_signals(self, user_id: str) -> dict:
        """
        Fraud signals for a user's audio. Same shape as SMS/email signals
        so Context can drop it into the bundle as 'audio_fraud_signals'.
        """
        transcripts = self._by_user_id.get(user_id, [])
        if not transcripts:
            return {
                "phishing_hits": 0, "suspicious_domains": [],
                "fraud_keywords": [], "voice_fraud_keywords": [],
            }

        # Aggregate across all transcripts for this user
        total_phishing = 0
        all_domains: set[str] = set()
        all_fraud_kw: set[str] = set()
        all_voice_kw: set[str] = set()

        for t in transcripts:
            sig = t.fraud_signals
            total_phishing += sig.get("phishing_hits", 0)
            all_domains.update(sig.get("suspicious_domains", []))
            all_fraud_kw.update(sig.get("fraud_keywords", []))
            all_voice_kw.update(sig.get("voice_fraud_keywords", []))

        return {
            "phishing_hits": total_phishing,
            "suspicious_domains": list(all_domains),
            "fraud_keywords": list(all_fraud_kw),
            "voice_fraud_keywords": list(all_voice_kw),
        }

    def fraud_signals_by_name(self, name: str) -> dict:
        """Fraud signals by speaker name. Fallback when user_id isn't known."""
        transcripts = self._by_name.get(name.lower().strip(), [])
        if not transcripts:
            return {
                "phishing_hits": 0, "suspicious_domains": [],
                "fraud_keywords": [], "voice_fraud_keywords": [],
            }

        total_phishing = 0
        all_domains: set[str] = set()
        all_fraud_kw: set[str] = set()
        all_voice_kw: set[str] = set()

        for t in transcripts:
            sig = t.fraud_signals
            total_phishing += sig.get("phishing_hits", 0)
            all_domains.update(sig.get("suspicious_domains", []))
            all_fraud_kw.update(sig.get("fraud_keywords", []))
            all_voice_kw.update(sig.get("voice_fraud_keywords", []))

        return {
            "phishing_hits": total_phishing,
            "suspicious_domains": list(all_domains),
            "fraud_keywords": list(all_fraud_kw),
            "voice_fraud_keywords": list(all_voice_kw),
        }

    # -- internal -----------------------------------------------------------

    def _transcribe_file(self, filepath: Path) -> Transcript | None:
        """Transcribe a single file and build a Transcript object."""
        metadata = parse_audio_filename(filepath)
        result = self._backend.transcribe(filepath)

        transcript = Transcript(
            filepath=str(filepath),
            speaker_name=metadata["speaker_name"],
            timestamp=metadata["datetime"],
            text=result["text"],
            language=result["language"],
            duration_seconds=result["duration"],
            fraud_signals=scan_transcript_for_fraud(result["text"]),
        )

        logger.debug(
            f"STT: {filepath.name} → {len(result['text'])} chars, "
            f"lang={result['language']}, "
            f"fraud_signals={transcript.fraud_signals['phishing_hits']} hits"
        )
        return transcript

    def _index_transcript(self, transcript: Transcript) -> None:
        """Index a transcript by speaker name and (if resolvable) user_id."""
        self._all_transcripts.append(transcript)

        # Index by speaker name
        name_key = transcript.speaker_name.lower().strip()
        if name_key not in self._by_name:
            self._by_name[name_key] = []
        self._by_name[name_key].append(transcript)

        # Try to resolve to a user_id
        user_id = match_speaker_to_user(
            transcript.speaker_name, self._users, self._sender_ids
        )
        if user_id:
            if user_id not in self._by_user_id:
                self._by_user_id[user_id] = []
            self._by_user_id[user_id].append(transcript)
            logger.debug(
                f"STT: matched '{transcript.speaker_name}' → {user_id}"
            )
        else:
            logger.debug(
                f"STT: no user match for speaker '{transcript.speaker_name}'"
            )
