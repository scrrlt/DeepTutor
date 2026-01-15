#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codebase Fixer Script
=====================

Automates fixing common code issues in the DeepTutor codebase:
1. Replace print statements with proper logging
2. Fix type annotation errors

Usage:
    python fix_codebase.py [--dry-run] [--fix-prints] [--fix-types]

Options:
    --dry-run: Show what would be changed without making changes
    --fix-prints: Replace print statements with logging calls
    --fix-types: Fix type annotation errors (requires mypy)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Set, Tuple


class CodeFixer:
    """Main class for fixing codebase issues"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"

        # Files to skip for print replacement (CLI tools, scripts, examples)
        self.skip_print_files = {
            "src/knowledge/start_kb.py",
            "src/knowledge/example_add_documents.py",
            "src/tools/",
            "src/scripts/",
            "src/agents/research/main.py",
            "src/agents/question/example.py",
            "src/knowledge/manager.py",  # CLI parts
        }

        # Files that already have logging
        self.files_with_logging = self._find_files_with_logging()

    def _find_files_with_logging(self) -> Set[str]:
        """Find files that already import logging"""
        files_with_logging = set()
        for root, dirs, files in os.walk(self.src_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                            if "from src.logging import" in content or "import logging" in content:
                                files_with_logging.add(filepath)
                    except Exception:
                        pass
        return files_with_logging

    def _should_skip_print_replacement(self, filepath: str) -> bool:
        """Check if file should be skipped for print replacement"""
        rel_path = os.path.relpath(filepath, self.project_root)
        # Normalize path separators for cross-platform compatibility
        rel_path = rel_path.replace("\\", "/")

        # Skip CLI tools and scripts
        for skip_pattern in self.skip_print_files:
            # Normalize skip pattern too
            skip_pattern = skip_pattern.replace("\\", "/")
            if rel_path.startswith(skip_pattern) or skip_pattern in rel_path:
                return True

        # Skip files that are mainly CLI
        if any(keyword in rel_path for keyword in ["main.py", "cli", "command"]):
            return True

        return False

    def _determine_log_level(self, print_content: str) -> str:
        """Determine appropriate log level for print content"""
        content_lower = print_content.lower()

        if any(word in content_lower for word in ["error", "failed", "exception", "err"]):
            return "error"
        elif any(word in content_lower for word in ["warning", "warn"]):
            return "warning"
        elif any(word in content_lower for word in ["debug"]):
            return "debug"
        elif any(word in content_lower for word in ["success", "completed"]):
            return "info"
        else:
            return "info"

    def _add_logging_to_file(self, content: str, filepath: str) -> str:
        """Add logging import and logger initialization to file"""
        lines = content.split("\n")

        # Check if already has logging
        if "from src.logging import" in content or "import logging" in content:
            return content

        # Check if get_logger is already imported
        if "get_logger" in content:
            return content

        # Find import section
        import_end = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_end = i + 1
            elif (
                line.strip()
                and not line.startswith("#")
                and not line.startswith('"""')
                and not line.startswith("'''")
            ):
                break

        # Add logging import
        if import_end > 0:
            lines.insert(import_end, "")
            lines.insert(import_end + 1, "from src.logging import get_logger")
            lines.insert(import_end + 2, "")

            # Add logger initialization after docstring if present
            logger_inserted = False
            in_docstring = False
            docstring_delimiter = None

            for i in range(import_end + 3, len(lines)):
                line = lines[i].strip()
                if not in_docstring and (line.startswith('"""') or line.startswith("'''")):
                    in_docstring = True
                    docstring_delimiter = line[:3]
                elif in_docstring and docstring_delimiter and line.endswith(docstring_delimiter):
                    in_docstring = False
                    # Insert logger after docstring
                    lines.insert(i + 1, f"logger = get_logger(__name__)")
                    lines.insert(i + 2, "")
                    logger_inserted = True
                    break
                elif not in_docstring and not line.startswith("#") and line:
                    # Insert before first non-comment, non-empty line
                    lines.insert(i, f"logger = get_logger(__name__)")
                    lines.insert(i + 1, "")
                    logger_inserted = True
                    break

            if not logger_inserted:
                lines.insert(import_end + 3, f"logger = get_logger(__name__)")
                lines.insert(import_end + 4, "")

        return "\n".join(lines)

    def _is_inside_string_or_comment(self, content: str, position: int) -> bool:
        """Check if a position in the content is inside a string literal or comment"""
        # Simple check: look backwards for unclosed quotes or comment markers
        before = content[:position]

        # Check for comments
        if "#" in before:
            # Find the last newline before position
            last_newline = before.rfind("\n")
            if last_newline == -1:
                last_newline = 0
            # Check if there's a # between last newline and position
            line_before = before[last_newline:position]
            if "#" in line_before:
                return True

        # Check for string literals (simplified - doesn't handle escaped quotes perfectly)
        single_quotes = before.count("'") - before.count("\\'")
        double_quotes = before.count('"') - before.count('\\"')

        return (single_quotes % 2 == 1) or (double_quotes % 2 == 1)

    def _replace_prints_in_file(self, filepath: str) -> Tuple[bool, int]:
        """Replace print statements in a single file"""
        if self._should_skip_print_replacement(filepath):
            return False, 0

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Add logging if not present
            content = self._add_logging_to_file(content, filepath)

            # Replace print statements - improved regex to handle more cases
            # This pattern matches print( followed by balanced parentheses content
            print_pattern = r"(\s*)print\((.*?)\)(?=\s*(?:#.*)?(?:\n|$))"
            replacements = 0

            def replace_print(match):
                nonlocal replacements
                indent = match.group(1)
                args = match.group(2).strip()

                # Skip if this looks like it's inside a string or comment
                if self._is_inside_string_or_comment(content, match.start()):
                    return match.group(0)

                log_level = self._determine_log_level(args)
                replacements += 1
                return f"{indent}logger.{log_level}({args})"

            content = re.sub(print_pattern, replace_print, content, flags=re.MULTILINE | re.DOTALL)

            if content != original_content and not self.dry_run:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(
                    f"Fixed {replacements} print statements in {os.path.relpath(filepath, self.project_root)}"
                )
                return True, replacements
            elif content != original_content and self.dry_run:
                print(
                    f"Would fix {replacements} print statements in {os.path.relpath(filepath, self.project_root)}"
                )
                return True, replacements

        except (IOError, OSError) as e:
            print(f"Error processing {filepath}: {e}")
            return False, 0

        return False, 0

    def fix_print_statements(self) -> None:
        """Fix print statements across the codebase"""
        print("Finding and fixing print statements...")

        total_files = 0
        total_prints = 0

        for root, _, files in os.walk(self.src_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    changed, count = self._replace_prints_in_file(filepath)
                    if changed:
                        total_files += 1
                        total_prints += count

        if self.dry_run:
            print(f"Would fix {total_prints} print statements in {total_files} files")
        else:
            print(f"Fixed {total_prints} print statements in {total_files} files")

    def fix_type_annotations(self) -> None:
        """Fix type annotation errors using mypy"""
        print("Running mypy to find type annotation issues...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "src",
                    "--ignore-missing-imports",
                    "--no-error-summary",
                    "--show-error-codes",
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                print("No type annotation errors found")
                return

            errors = [
                line for line in result.stdout.split("\n") if "error:" in line and "src/" in line
            ]
            print(f"Found {len(errors)} type annotation errors")

            if self.dry_run:
                print("Would fix type annotation errors (not implemented yet)")
            else:
                print("Type annotation fixing not fully implemented yet")
                print("Manual review recommended for mypy errors")

            for error in errors[:5]:
                print(f"  {error}")

            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")

        except FileNotFoundError:
            print("mypy not installed. Install with: pip install mypy")
        except (subprocess.SubprocessError, OSError) as e:
            print(f"Error running mypy: {e}")

    def run(self, fix_prints: bool = True, fix_types: bool = True) -> None:
        """Run all fixes"""
        print("Starting codebase fixes...")
        print(f"Project root: {self.project_root}")
        print(f"Source directory: {self.src_dir}")
        print(
            f"Total Python files found: {sum(1 for _, _, files in os.walk(self.src_dir) for file in files if file.endswith('.py'))}"
        )

        if self.dry_run:
            print("Running in dry-run mode (no changes will be made)")

        if fix_prints:
            self.fix_print_statements()

        if fix_types:
            self.fix_type_annotations()

        print("Codebase fixes completed!")


def main():
    parser = argparse.ArgumentParser(description="Fix common codebase issues")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--fix-prints", action="store_true", help="Replace print statements with logging calls"
    )
    parser.add_argument("--fix-types", action="store_true", help="Fix type annotation errors")

    args = parser.parse_args()

    if not args.fix_prints and not args.fix_types:
        args.fix_prints = True
        args.fix_types = True

    fixer = CodeFixer(dry_run=args.dry_run)
    fixer.run(fix_prints=args.fix_prints, fix_types=args.fix_types)


if __name__ == "__main__":
    main()
