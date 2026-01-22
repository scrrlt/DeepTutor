import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def audit_filenames(root_dir: str) -> int:
    """Audit filenames for potential encoding issues.

    Returns the number of corrupted entries found.
    """
    logger.info("Starting audit of: %s", root_dir)
    corrupted_found = 0
    total_files = 0

    for root, dirs, files in os.walk(root_dir):
        # Check directories and files for non-ASCII or mojibake
        for name in dirs + files:
            total_files += 1
            full_path = os.path.join(root, name)

            try:
                # Test if the name can be encoded/decoded cleanly
                name.encode("ascii")
            except UnicodeEncodeError:
                logger.warning("Potential corruption: %s", full_path)
                corrupted_found += 1
            except Exception as e:
                logger.error("Could not process: %s (%s)", full_path, e)
                corrupted_found += 1

    logger.info("Audit complete")
    logger.info("Total entries scanned: %d", total_files)
    logger.info("Corrupted entries found: %d", corrupted_found)

    if corrupted_found == 0:
        logger.info("Filesystem naming looks clean.")
    else:
        logger.warning("Corruption detected. Run 'git clean -fd' again.")

    return corrupted_found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit filenames for encoding issues"
    )
    parser.add_argument(
        "root", nargs="?", default=os.getcwd(), help="Root directory to audit"
    )
    args = parser.parse_args()

    rc = audit_filenames(args.root)
    sys.exit(0 if rc == 0 else 2)
