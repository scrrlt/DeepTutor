from collections.abc import Generator
import os
import re

import tiktoken

from src.logging import get_logger

logger = get_logger("TexChunker")


class TokenBuffer:
    """
    Accumulates text until a token limit is reached, then yields chunks
    with perfect token-level overlap handling.
    """

    def __init__(self, encoder: tiktoken.Encoding, max_tokens: int, overlap: int):
        self.encoder = encoder
        self.max_tokens = max_tokens
        self.overlap = overlap
        self._buffer_text: list[str] = []
        self._buffer_tokens = 0

    def add_text(self, text: str) -> Generator[str, None, None]:
        """
        Add text to buffer. Yields chunks if limit exceeded.
        """
        if not text:
            return

        text_tokens = len(self.encoder.encode(text))

        # If adding this text exceeds limit, we might need to flush
        if self._buffer_tokens + text_tokens > self.max_tokens:
            # If the buffer is not empty, flush it first
            if self._buffer_text:
                yield self._flush()

            # If the new text ITSELF is bigger than max_tokens,
            # we must split it aggressively (this handles the "giant paragraph" case)
            if text_tokens > self.max_tokens:
                yield from self._split_large_text(text)
                return

        self._buffer_text.append(text)
        self._buffer_tokens += text_tokens

    def _flush(self) -> str:
        """Combine buffer, return chunk, and set up overlap for next chunk."""
        full_text = "".join(self._buffer_text)

        # Calculate overlap for the NEXT buffer
        # We decode the last N tokens of the current text to be the start of the next
        tokens = self.encoder.encode(full_text)
        if len(tokens) > self.overlap:
            overlap_tokens = tokens[-self.overlap :]
            overlap_text = self.encoder.decode(overlap_tokens)
            self._buffer_text = [overlap_text]
            self._buffer_tokens = len(overlap_tokens)
        else:
            self._buffer_text = []
            self._buffer_tokens = 0

        return full_text

    def _split_large_text(self, text: str) -> Generator[str, None, None]:
        """Fallback for atomic units larger than context window."""
        tokens = self.encoder.encode(text)
        for i in range(0, len(tokens), self.max_tokens - self.overlap):
            chunk_tokens = tokens[i : i + self.max_tokens]
            yield self.encoder.decode(chunk_tokens)

    def finish(self) -> Generator[str, None, None]:
        """Yield the remaining content in the buffer."""
        if self._buffer_text:
            yield "".join(self._buffer_text)


_ENCODER_CACHE: dict[str | None, tiktoken.Encoding] = {}


def _get_encoder(model: str | None) -> tiktoken.Encoding:
    """Return a cached tiktoken encoder for the requested model."""
    key = model or "__default__"
    cached = _ENCODER_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        if model:
            encoder = tiktoken.encoding_for_model(model)
        else:
            encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")

    _ENCODER_CACHE[key] = encoder
    return encoder


class TexChunker:
    """LaTeX text chunking tool"""

    def __init__(self, model: str | None = None):
        """
        Initialize chunking tool

        Args:
            model: Model name (for token estimation). If not provided, read from LLM_MODEL environment variable
        """
        # Read model configuration from environment variables
        if model is None:
            model = os.getenv("LLM_MODEL")
        self.encoder = _get_encoder(model)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count of text

        Args:
            text: Input text

        Returns:
            Token count
        """
        try:
            tokens = self.encoder.encode(text)
            return len(tokens)
        except Exception as e:
            # If encoding fails, use rough estimate: 1 token ≈ 4 chars
            logger.warning("Token estimation failed, using rough estimate: %s", e)
            logger.warning("Token estimation failed, using rough estimate: %s", e)
            return len(text) // 4

    def split_tex_into_chunks(
        self, tex_content: str, max_tokens: int = 8000, overlap: int = 500
    ) -> list[str]:
        r"""
        Split LaTeX content into chunks

        Strategy:
        1. Prioritize splitting by sections (\section, \subsection)
        2. If single section is too long, split by paragraphs
        3. Maintain overlap tokens to avoid context loss

        Args:
            tex_content: LaTeX source code
            max_tokens: Maximum tokens per chunk (default: 8000)
            overlap: Overlap tokens between chunks (default: 500)

        Returns:
            List of chunks
        """
        buffer = TokenBuffer(self.encoder, max_tokens, overlap)
        chunks = []

        # 1. Split by logical LaTeX sections
        # Use a non-greedy regex that tolerates some nesting but relies on newlines
        # assuming logical breaks usually happen at line starts in source code.
        section_pattern = r"(^|\n)(\\(?:sub)*section\*?\{.*?\})"

        parts = re.split(section_pattern, tex_content, flags=re.MULTILINE)

        # Recombine: merge section markers and content
        current_block = ""

        # re.split with groups returns [text, prefix, delimiter, text...]
        # parts index: 0=preamble, 1=newline/start, 2=delimiter, 3=content...

        # Initial preamble (parts[0])
        if parts[0].strip():
            chunks.extend(buffer.add_text(parts[0]))

        i = 1
        while i < len(parts):
            prefix = parts[i] or ""
            delimiter = parts[i + 1] if i + 1 < len(parts) else ""
            content = parts[i + 2] if i + 2 < len(parts) else ""

            section_full = prefix + delimiter

            # Level 2: Split by Paragraphs within the section
            paragraphs = re.split(r"\n\n+", content)

            # Combine section header with first paragraph
            if paragraphs:
                paragraphs[0] = section_full + paragraphs[0]

            for para in paragraphs:
                if not para.strip():
                    continue
                # Re-attach the double newline we lost
                clean_para = para + "\n\n"
                chunks.extend(buffer.add_text(clean_para))

            i += 3

        chunks.extend(buffer.finish())

        logger.info("Chunking completed: %d chunks", len(chunks))
        return chunks


# ========== Usage Example ==========

if __name__ == "__main__":
    # Create chunking tool
    chunker = TexChunker(model="gpt-4o")

    # Test text
    test_tex = r"""
\section{Introduction}
This is the introduction section with some content that is moderately long.
It contains multiple paragraphs and discusses the background of the research.

The problem we are addressing is important and has wide applications.

\section{Related Work}
Previous work has explored various approaches to this problem.
Some researchers have used method A, while others prefer method B.

Recent advances in deep learning have opened new possibilities.

\subsection{Deep Learning Approaches}
Neural networks have shown promising results in many tasks.
Convolutional networks are particularly effective for image processing.

\section{Methodology}
Our approach combines the best aspects of previous methods.
We propose a novel architecture that addresses the key limitations.

\subsection{Model Architecture}
The model consists of three main components: encoder, processor, and decoder.
Each component is carefully designed to handle specific aspects of the task.

\section{Experiments}
We conducted extensive experiments on multiple datasets.
The results demonstrate the effectiveness of our approach.
    """

    # Estimate tokens
    total_tokens = chunker.estimate_tokens(test_tex)
    logger.info("Total tokens: %d", total_tokens)

    # Chunk (set smaller max_tokens for demonstration)
    chunks = chunker.split_tex_into_chunks(tex_content=test_tex, max_tokens=200, overlap=50)

    logger.info("Chunking result: %d chunks", len(chunks))

    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = chunker.estimate_tokens(chunk)
        logger.info("Chunk %d (%d tokens):", i, chunk_tokens)
        logger.debug(chunk[:200] + "...\n")
