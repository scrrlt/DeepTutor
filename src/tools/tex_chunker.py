# -*- coding: utf-8 -*-
"""
TeX Chunker - LaTeX text chunking tool

Features:
1. Intelligent chunking of LaTeX content (by section or token count)
2. Token estimation (based on GPT tokenizer)
3. Maintain context coherence (overlap between chunks)

Author: DeepTutor Team
Version: v1.0
Based on: TODO.md specification
"""

import logging
import os
import re

import tiktoken

logger = logging.getLogger(__name__)


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

        try:
            if model:
                self.encoder = tiktoken.encoding_for_model(model)
            else:
                # Use cl100k_base as default encoding if no model specified
                self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # If model not supported, use cl100k_base (GPT-4 encoding)
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the provided text.
        
        The text is normalized via the chunker's cleaning routine before encoding. If encoding succeeds, returns the count of encoded tokens; if encoding fails, logs a warning and returns a rough estimate calculated as len(text) // 4.
        
        Parameters:
            text (str): Input text to estimate tokens for; will be cleaned before tokenization.
        
        Returns:
            int: Estimated token count for the input text (or a rough estimate on encoding failure).
        """
        try:
            cleaned_text = self._clean_text(text)
            tokens = self.encoder.encode(cleaned_text)
            return len(tokens)
        except Exception as e:
            # If encoding fails, use rough estimate: 1 token ≈ 4 chars
            logger.warning("Token estimation failed, using rough estimate: %s", e)
            return len(text) // 4

    def _clean_text(self, text: str) -> str:
        """
        Normalize input text to reduce tokenization anomalies.
        
        This function collapses sequences of the same character longer than ten to at most two repeats, replaces any run of whitespace (spaces, newlines, tabs) with a single space, and trims leading/trailing whitespace. The result is intended for more stable token-count estimation.
        
        Parameters:
            text (str): The input text to normalize.
        
        Returns:
            str: The normalized text with long character repeats collapsed, whitespace normalized, and edges trimmed.
        """
        import re

        # Remove overly long repeated characters (e.g., consecutive spaces, newlines, etc.)
        text = re.sub(r'(.)\1{10,}', r'\1\1', text)  # Collapse sequences longer than 10
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

    def split_tex_into_chunks(
        self, tex_content: str, max_tokens: int = 8000, overlap: int = 500
    ) -> list[str]:
        """
        Split LaTeX source into token-bounded chunks with optional overlap.
        
        Chunks are produced to respect the provided token limit where possible; section boundaries are preferred, and overly long sections are further split by paragraphs. Consecutive chunks may include an overlap of trailing tokens from the previous chunk to preserve context.
        
        Parameters:
            tex_content (str): LaTeX source to split.
            max_tokens (int): Target maximum tokens per chunk.
            overlap (int): Number of tokens to include from the end of the previous chunk at the start of the next chunk to preserve context.
        
        Returns:
            list[str]: List of LaTeX chunk strings; chunks are sized to avoid exceeding `max_tokens` where possible and include `overlap` tokens between adjacent chunks.
        """
        total_tokens = self.estimate_tokens(tex_content)

        # If total length doesn't exceed max_tokens, return directly
        if total_tokens <= max_tokens:
            return [tex_content]

        logger.info(
            "LaTeX content needs chunking: %s tokens > %s tokens",
            f"{total_tokens:,}",
            f"{max_tokens:,}",
        )

        logger.info(
            f"File character count: {len(tex_content):,}, line count: {len(tex_content.splitlines()):,}"
        )

        # 1. Try splitting by sections
        sections = self._split_by_sections(tex_content)

        # 2. Merge sections into chunks
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for section in sections:
            section_tokens = self.estimate_tokens(section)

            if section_tokens > max_tokens:
                # Single section too long, need further splitting
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                # Split overly long section by paragraphs
                sub_chunks = self._split_by_paragraphs(section, max_tokens, overlap)
                chunks.extend(sub_chunks)
            # Check if can merge into current chunk
            elif current_tokens + section_tokens <= max_tokens:
                current_chunk += section
                current_tokens += section_tokens
            else:
                # Save current chunk, start new chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # Add overlap (take part from end of current chunk)
                if chunks and overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1], overlap)
                    current_chunk = overlap_text + section
                    current_tokens = self.estimate_tokens(current_chunk)
                else:
                    current_chunk = section
                    current_tokens = section_tokens

        # Save last chunk
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Chunking completed: {len(chunks)} chunks")
        return chunks

    def _split_by_sections(self, tex_content: str) -> list[str]:
        """
        Split LaTeX content into top-level section blocks.
        
        This function partitions the input TeX text at LaTeX sectioning commands (\section{...}, \subsection{...}, \subsubsection{...}) and returns each section marker together with its following content as a block. If no section markers are found, the function falls back to splitting the content into paragraph-based chunks via the paragraph-splitting helper.
        
        Parameters:
            tex_content (str): Raw LaTeX document text to split.
        
        Returns:
            list[str]: A list of text blocks where each element is either a preamble (text before the first section) or a section block beginning with a sectioning command followed by its content.
        """
        # Regex match section markers
        pattern = r"(\\(?:sub)*section\{[^}]*\})"

        # Split text
        parts = re.split(pattern, tex_content)

        if len(parts) <= 1:
            # No section markers found, split by paragraphs
            return self._split_by_paragraphs(tex_content, max_tokens=10000, overlap=0)

        # Recombine: merge section markers and content
        sections = []
        for i in range(1, len(parts), 2):
            if i < len(parts):
                section = parts[i]  # Section marker
                if i + 1 < len(parts):
                    section += parts[i + 1]  # Section content
                sections.append(section)

        # Add preamble part (first element)
        if parts[0].strip():
            sections.insert(0, parts[0])

        return sections

    def _split_by_paragraphs(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """
        Split text by paragraphs (for overly long sections)

        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk
            overlap: Overlap tokens

        Returns:
            List of paragraph chunks
        """
        # Split paragraphs by double newlines
        paragraphs = re.split(r"\n\n+", text)

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)

            if para_tokens > max_tokens:
                # Single paragraph too long, split by sentences
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                # Split by sentences (simple method: split by periods)
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    sentence_tokens = self.estimate_tokens(sentence)
                    if current_tokens + sentence_tokens <= max_tokens:
                        current_chunk += sentence + " "
                        current_tokens += sentence_tokens
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence + " "
                        current_tokens = sentence_tokens
            # Check if can merge
            elif current_tokens + para_tokens <= max_tokens:
                current_chunk += para + "\n\n"
                current_tokens += para_tokens
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # Add overlap
                if chunks and overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1], overlap)
                    current_chunk = overlap_text + para + "\n\n"
                    current_tokens = self.estimate_tokens(current_chunk)
                else:
                    current_chunk = para + "\n\n"
                    current_tokens = para_tokens

        # Save last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _get_overlap_text(self, previous_chunk: str, overlap_tokens: int) -> str:
        """
        Extract overlap portion from end of previous chunk

        Args:
            previous_chunk: Previous chunk
            overlap_tokens: Number of overlap tokens

        Returns:
            Overlap text
        """
        # Encode entire chunk
        tokens = self.encoder.encode(previous_chunk)

        # Take last overlap_tokens tokens
        if len(tokens) <= overlap_tokens:
            return previous_chunk

        overlap_token_ids = tokens[-overlap_tokens:]
        overlap_text = self.encoder.decode(overlap_token_ids)

        return overlap_text


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
    print(f"Total tokens: {total_tokens}")

    # Chunk (set smaller max_tokens for demonstration)
    chunks = chunker.split_tex_into_chunks(tex_content=test_tex, max_tokens=200, overlap=50)

    print(f"\nChunking result: {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = chunker.estimate_tokens(chunk)
        print(f"Chunk {i} ({chunk_tokens} tokens):")
        print(chunk[:200] + "...\n")