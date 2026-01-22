#!/usr/bin/env python
"""
Parse PDF files using MinerU with atomic file operations and error handling.
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys
import tempfile

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PDFParser")


def _get_mineru_command() -> str | None:
    """
    Resolve the MinerU executable using shutil.which for speed and safety.
    """
    if shutil.which("magic-pdf"):
        return "magic-pdf"
    if shutil.which("mineru"):
        return "mineru"
    return None


def parse_pdf_with_mineru(
    pdf_path: str | Path,
    output_base_dir: str | Path | None = None,
) -> bool:
    """
    Parse PDF using MinerU with atomic temporary directory handling.

    Args:
        pdf_path: Path to the source PDF.
        output_base_dir: Destination directory.

    Returns:
        bool: Success status.
    """
    mineru_cmd = _get_mineru_command()
    if not mineru_cmd:
        logger.error(
            "MinerU not found. Install via: pip install magic-pdf[full] or pip install mineru"
        )
        return False

    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.exists() or pdf_file.suffix.lower() != ".pdf":
        logger.error(f"Invalid PDF file: {pdf_file}")
        return False

    # Default to 'reference_papers' in CWD if not specified
    if output_base_dir:
        try:
            output_root = Path(output_base_dir).resolve()
        except (TypeError, ValueError, OSError) as e:
            logger.error("Invalid output directory %r: %s", output_base_dir, e)
            return False
    else:
        # Better than parent.parent... assume relative to execution context or specific env var
        output_root = Path.cwd() / "reference_papers"

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_root}: {e}")
        return False

    final_dest_dir = output_root / pdf_file.stem

    # Handle existing directory with rotation
    if final_dest_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = output_root / f"{pdf_file.stem}_backup_{timestamp}"
        logger.warning(f"Output exists. Rotating old data to: {backup_dir.name}")
        try:
            shutil.move(str(final_dest_dir), str(backup_dir))
        except OSError as e:
            logger.error(f"Failed to backup existing directory: {e}")
            return False

    logger.info(f"Processing: {pdf_file}")

    # defensive: Use TemporaryDirectory for atomic cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        cmd = [mineru_cmd, "-p", str(pdf_file), "-o", str(temp_path)]
        logger.debug(f"Exec: {' '.join(cmd)}")

        try:
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess failed to launch: {e}")
            return False

        if result.returncode != 0:
            logger.error(f"MinerU failed (Exit {result.returncode})")
            logger.error(f"Stderr: {result.stderr}")
            return False

        # MinerU creates a subdirectory named after the PDF inside the output dir
        # We need to find it dynamically rather than assuming list index [0]
        expected_subdir = temp_path / pdf_file.stem

        # Fallback: scan for any directory if the expected one isn't found
        if not expected_subdir.exists():
            subdirs = [x for x in temp_path.iterdir() if x.is_dir()]
            if not subdirs:
                logger.error("MinerU reported success but produced no output directories.")
                return False
            expected_subdir = subdirs[0]

        try:
            # Atomic move from temp to final
            shutil.move(str(expected_subdir), str(final_dest_dir))
            logger.info(f"Success. Output saved to: {final_dest_dir}")

            # Log produced files
            for file in final_dest_dir.rglob("*"):
                if file.is_file():
                    logger.debug(f"Generated: {file.name}")

            return True

        except OSError as e:
            logger.error(f"Failed to move files from temp to destination: {e}")
            return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust MinerU PDF Parser",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("pdf_path", help="Path to source PDF")
    parser.add_argument("-o", "--output", help="Output directory root")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Also ensure the root handler allows DEBUG messages
        for handler in logging.root.handlers:
            handler.setLevel(logging.DEBUG)

    if not parse_pdf_with_mineru(args.pdf_path, args.output):
        sys.exit(1)


if __name__ == "__main__":
    main()
