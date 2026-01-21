# -*- coding: utf-8 -*-
"""Streaming parser utilities for handling partial thinking markers across chunks."""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.services.llm.utils import clean_thinking_tags
from src.services.llm.capabilities import get_thinking_markers


class StreamParser:
    """Stateful parser that buffers stream chunks and emits safe output.

    Behavior:
    - Buffers incoming text until a complete open/close marker pair is seen
    - Avoids emitting partial markers (e.g., "<thi") to prevent leaking tags
    - Supports multiple marker styles: "<think>...</think>", "◣...◢", "꽁...꽁"

    Args:
        binding: Provider binding name (e.g., "openai", "local")
        model: Model name (optional)
    """

    DEFAULT_OPEN_MARKERS: tuple[str, ...] = ("<think>", "◣", "꽁")
    DEFAULT_CLOSE_MARKERS: tuple[str, ...] = ("</think>", "◢", "꽁")

    def __init__(self, binding: Optional[str] = None, model: Optional[str] = None):
        self.binding = binding
        self.model = model

        # Buffers
        self.yield_buffer: str = ""  # content awaiting emission or pre-marker text
        self.thinking_buffer: str = ""
        self.in_thinking_block: bool = False
        self._pre_marker: str | None = None  # holds pre-marker text until the thinking block closes

        # Resolve markers from capabilities (provider/model), fall back to defaults
        open_markers, close_markers = get_thinking_markers(binding, model)
        # Normalize order: prefer longer markers first to avoid prefix collisions
        self.open_markers = (
            tuple(sorted(open_markers, key=len, reverse=True)) or self.DEFAULT_OPEN_MARKERS
        )
        self.close_markers = (
            tuple(sorted(close_markers, key=len, reverse=True)) or self.DEFAULT_CLOSE_MARKERS
        )

    def _ends_with_partial(self, buf: str, markers: Tuple[str, ...]) -> bool:
        for m in markers:
            for k in range(1, len(m)):
                if buf.endswith(m[:k]):
                    return True
        return False

    def _find_first_marker(self, buf: str, markers: Tuple[str, ...]):
        first_idx = None
        first_marker = None
        for m in markers:
            idx = buf.find(m)
            if idx != -1 and (first_idx is None or idx < first_idx):
                first_idx = idx
                first_marker = m
        return first_idx, first_marker

    def append(self, content: str) -> List[str]:
        """Append a new chunk and return any completed outputs."""
        outputs: List[str] = []
        self.yield_buffer += content

        progressed = True
        while progressed:
            progressed = False

            if not self.in_thinking_block:
                idx, _ = self._find_first_marker(self.yield_buffer, self.open_markers)

                if self._ends_with_partial(self.yield_buffer, self.open_markers):
                    break  # wait for more data

                if idx is not None:
                    # Found a full start marker - move the thinking block into a buffer
                    # and hold pre-marker text in _pre_marker until a close marker is found.
                    self._pre_marker = self.yield_buffer[:idx] if idx > 0 else ""
                    self.thinking_buffer = self.yield_buffer[idx:]
                    self.yield_buffer = ""
                    self.in_thinking_block = True

                    # If a close is already present in thinking_buffer, process immediately
                    close_idx, close_marker = self._find_first_marker(
                        self.thinking_buffer, self.close_markers
                    )
                    if close_idx is not None:
                        # When a close marker is found immediately, clean and emit
                        # the pre-marker text and the trailing 'after' content.
                        after = self.thinking_buffer.split(close_marker, 1)[1]
                        _ = clean_thinking_tags(self.thinking_buffer, self.binding, self.model)
                        self.thinking_buffer = ""
                        self.in_thinking_block = False
                        pre = self._pre_marker or ""
                        self._pre_marker = None
                        if pre:
                            outputs.append(pre)
                        if after:
                            outputs.append(after)
                        progressed = True
                        continue
                else:
                    # No start marker present; keep buffer until finalization
                    # (do not emit partial text here to avoid leaking partial tags).
                    pass
            else:
                # inside thinking block
                self.thinking_buffer += self.yield_buffer
                self.yield_buffer = ""

                if self._ends_with_partial(self.thinking_buffer, self.close_markers):
                    break  # wait for more data

                close_idx, close_marker = self._find_first_marker(
                    self.thinking_buffer, self.close_markers
                )
                if close_idx is not None:
                    # Found closing marker while inside thinking block.
                    # Clean the thinking buffer and emit the pre_marker followed by
                    # the trailing after content immediately.
                    after = self.thinking_buffer.split(close_marker, 1)[1]
                    _ = clean_thinking_tags(self.thinking_buffer, self.binding, self.model)
                    self.thinking_buffer = ""
                    self.in_thinking_block = False
                    pre = self._pre_marker or ""
                    self._pre_marker = None
                    if pre:
                        outputs.append(pre)
                    if after:
                        outputs.append(after)
                    progressed = True
                    continue
                # otherwise still inside thinking block, wait
                break

        return outputs

    def finalize(self) -> List[str]:
        """Flush any remaining buffered content at end-of-stream.

        If a thinking block was left open, attempt to recover by:
        1) Running clean_thinking_tags (may remove tags and reveal trailing text)
        2) If cleaning yields nothing, try stripping the leading open marker and
           presenting the remainder as best-effort recovery so the user sees
           the last partial output.
        """
        outputs: List[str] = []
        if self.in_thinking_block and self.thinking_buffer:
            cleaned = clean_thinking_tags(self.thinking_buffer, self.binding, self.model)
            if cleaned:
                outputs.append(cleaned)
            else:
                # Attempt to strip a leading open marker and yield remainder
                yielded = False
                for m in self.open_markers:
                    if self.thinking_buffer.startswith(m):
                        candidate = self.thinking_buffer[len(m) :]
                        if candidate:
                            outputs.append(candidate)
                            yielded = True
                            break
                if not yielded and self.thinking_buffer:
                    outputs.append(self.thinking_buffer)
        elif self.yield_buffer:
            outputs.append(self.yield_buffer)
        # Reset buffers
        self.yield_buffer = ""
        self.thinking_buffer = ""
        self.in_thinking_block = False
        return outputs
