# -*- coding: utf-8 -*-
"""
Streaming Parser (FSM Implementation)
=====================================

Robust, fail-closed finite state machine for stripping 'thinking' blocks
from LLM streams. Optimized for Python 3.12+.

Architecture:
    - State Machine: Explicit transitions between NORMAL and THINKING states.
    - String Searching: Uses .find() for performance (O(N) vs O(N*M)).
    - Fail-Closed: Unfinished thinking blocks are discarded, never leaked.
    - Memory Safety: Enforces hard limits on buffered thought size.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import List, Optional, Tuple

from src.services.llm.capabilities import get_thinking_markers

logger = logging.getLogger(__name__)


class ParseState(Enum):
    """Parser operational modes."""

    NORMAL = auto()  # Passing through content to user
    THINKING = auto()  # Suppressing/Buffering thought content


class StreamParser:
    """
    Finite State Machine parser for robust thinking tag removal.

    Attributes:
        MAX_THOUGHT_SIZE (int): Hard limit (bytes) for thought buffers (1MB).
                                Prevents DoS via infinite buffering.
    """

    # 1MB Safety Limit
    MAX_THOUGHT_SIZE = 1024 * 1024

    DEFAULT_OPEN_MARKERS: tuple[str, ...] = ("<think>", "◣", "꽁")
    DEFAULT_CLOSE_MARKERS: tuple[str, ...] = ("</think>", "◢", "꽁")

    def __init__(self, binding: Optional[str] = None, model: Optional[str] = None):
        self.binding = binding
        self.model = model
        self.state = ParseState.NORMAL

        # Operational Buffer
        # Holds content that is ambiguous (partial markers) or being scanned.
        self.buffer: str = ""

        # Stats for memory safety
        self.current_thought_size: int = 0

        # Configuration
        open_markers, close_markers = get_thinking_markers(binding, model)

        # Sort by length descending to prevent prefix collisions
        # e.g., match "<think>" before "<t"
        self.open_markers = tuple(
            sorted(open_markers or self.DEFAULT_OPEN_MARKERS, key=len, reverse=True)
        )
        self.close_markers = tuple(
            sorted(close_markers or self.DEFAULT_CLOSE_MARKERS, key=len, reverse=True)
        )

    def _find_first(
        self, text: str, markers: Tuple[str, ...]
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Find earliest occurrence of any marker using optimized string searching.
        Returns (index, marker) or (None, None).
        """
        best_idx: Optional[int] = None
        best_marker: Optional[str] = None

        for m in markers:
            idx = text.find(m)
            if idx != -1:
                if best_idx is None or idx < best_idx:
                    best_idx = idx
                    best_marker = m
        return best_idx, best_marker

    def _get_safe_length(self, text: str, markers: Tuple[str, ...]) -> int:
        """
        Calculate the length of the string prefix that is safe to emit.

        A suffix is 'unsafe' if it matches the START of any marker.
        Example: if marker is "<think>", text "hello <t" has unsafe suffix "<t".
        Safe length would be len("hello ").
        """
        if not text:
            return 0

        # Optimization: Only check tail equal to longest marker
        max_marker_len = max((len(m) for m in markers), default=0)
        check_len = min(len(text), max_marker_len)

        # Scan suffixes from longest to shortest.
        # We are looking for the longest suffix of `text` that is also a prefix
        # of any marker. Such a suffix may be the start of a marker that
        # continues in future chunks, so we must keep it in the buffer and only
        # emit the content before it.

        # Iterate backwards from the end of the string
        # Example: text "ABC<t", len=5. For max_marker_len=2, check range: 1..2
        for i in range(check_len, 0, -1):
            suffix = text[-i:]
            # Check if this suffix starts any marker
            for m in markers:
                if m.startswith(suffix):
                    # Potential split marker detected.
                    # Everything before this suffix is safe.
                    return len(text) - i

        return len(text)

    def append(self, content: str) -> List[str]:
        """
        Ingest chunks, update FSM state, and return safe output.
        """
        self.buffer += content
        outputs: List[str] = []

        # Process buffer until no state transitions or emissions are possible
        while True:
            if self.state == ParseState.NORMAL:
                # --- State: NORMAL ---
                # Goal: Find OPEN marker.
                # Action: Emit safe text, transition to THINKING if found.

                idx, marker = self._find_first(self.buffer, self.open_markers)

                if idx is not None:
                    # Case 1: Full Open Marker Found
                    # Emit everything before the marker
                    if idx > 0:
                        outputs.append(self.buffer[:idx])

                    # Transition State
                    self.state = ParseState.THINKING
                    self.current_thought_size = 0

                    # Consume buffer up to end of marker
                    # Note: We discard the open tag itself.
                    self.buffer = self.buffer[idx + len(marker) :]
                    continue  # Loop to check for immediate close in remainder

                else:
                    # Case 2: No Full Marker
                    # Check for partial markers at the end (e.g., "<th")
                    safe_len = self._get_safe_length(self.buffer, self.open_markers)

                    if safe_len < len(self.buffer):
                        # Partial match detected. Emit safe part, keep risky tail.
                        if safe_len > 0:
                            outputs.append(self.buffer[:safe_len])
                            self.buffer = self.buffer[safe_len:]
                        # Break to wait for next chunk to resolve ambiguity
                        break
                    else:
                        # No partials. Emit all, clear buffer.
                        if self.buffer:
                            outputs.append(self.buffer)
                            self.buffer = ""
                        break

            elif self.state == ParseState.THINKING:
                # --- State: THINKING ---
                # Goal: Find CLOSE marker.
                # Action: Buffer/Discard content, transition to NORMAL if found.

                idx, marker = self._find_first(self.buffer, self.close_markers)

                if idx is not None:
                    # Case 3: Full Close Marker Found
                    # The content before idx is the end of the thought.
                    thought_chunk_len = idx
                    # Soft limit check for auditing/monitoring only; content is discarded regardless.
                    if self.current_thought_size + thought_chunk_len > self.MAX_THOUGHT_SIZE:
                        logger.warning(
                            f"Thought block exceeded soft limit of {self.MAX_THOUGHT_SIZE} bytes "
                            f"before close marker. Content has been discarded; logging for audit only."
                            f"Truncating internal tracking."
                        )

                    # Transition State
                    self.state = ParseState.NORMAL
                    self.current_thought_size = 0

                    # Remove thought + close marker from buffer
                    self.buffer = self.buffer[idx + len(marker) :]
                    continue  # Loop to process potential normal text after tag

                else:
                    # Case 4: No Full Close Marker
                    # We must handle partial close markers (e.g., "</thi")
                    # We can safely discard everything that is NOT a partial close marker.

                    # Calculate how much we can safely discard into the void
                    safe_len = self._get_safe_length(self.buffer, self.close_markers)

                    discard_len = safe_len

                    # Enforce Memory Limit
                    if self.current_thought_size + discard_len > self.MAX_THOUGHT_SIZE:
                        # We are over limit. Just discard, don't increment counter significantly
                        # to prevent integer overflow in extreme edge cases (though Python handles large ints).
                        self.current_thought_size = self.MAX_THOUGHT_SIZE
                    else:
                        self.current_thought_size += discard_len

                    # Slice buffer: discard safe thought content, keep potential partial marker
                    if discard_len > 0:
                        self.buffer = self.buffer[discard_len:]

                    break  # Wait for more data

        return outputs

    def finalize(self) -> List[str]:
        """
        End of stream handler.

        FAIL-CLOSED POLICY:
        If we are still in THINKING state, the thought was never closed.
        We strictly DISCARD the buffer to prevent leaking internal reasoning.
        """
        outputs: List[str] = []

        if self.state == ParseState.THINKING:
            if self.current_thought_size > 0 or self.buffer:
                logger.warning(
                    f"Stream ended inside thinking block. "
                    f"Suppressed {self.current_thought_size + len(self.buffer)} bytes of unfinished thought."
                )
            # Do not emit buffer.
            self.buffer = ""
            self.state = ParseState.NORMAL
            self.current_thought_size = 0

        else:
            # Normal state: emit any leftovers
            if self.buffer:
                outputs.append(self.buffer)
                self.buffer = ""

        return outputs
