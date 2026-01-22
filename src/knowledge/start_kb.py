#!/usr/bin/env python
"""
Knowledge Base Management Startup Script - Unified Entry Point
Provides knowledge base initialization, management, querying, and other functions
"""

import argparse
import asyncio
from pathlib import Path
import sys

# Set paths - compatible with both direct execution and module import
try:
    from .config import KNOWLEDGE_BASES_DIR, get_env_config, setup_paths

    setup_paths()
    from src.services.rag.components.routing import FileTypeRouter

    from .extract_numbered_items import process_content_list
    from .initializer import KnowledgeBaseInitializer
    from .manager import KnowledgeBaseManager

except ImportError:
    # If relative import fails, means this file is run directly
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.knowledge.config import (
        KNOWLEDGE_BASES_DIR,
        get_env_config,
        setup_paths,
    )

    setup_paths()
    from src.knowledge.extract_numbered_items import process_content_list
    from src.knowledge.initializer import KnowledgeBaseInitializer
    from src.knowledge.manager import KnowledgeBaseManager
    from src.services.rag.components.routing import FileTypeRouter


from src.logging import get_logger

logger = get_logger("StartKB")


def list_knowledge_bases():
    """List all knowledge bases"""
    manager = KnowledgeBaseManager(str(KNOWLEDGE_BASES_DIR))
    kb_list = manager.list_knowledge_bases()
    default_kb = manager.get_default()

    logger.section("Available Knowledge Bases")

    if not kb_list:
        logger.warning("No knowledge bases yet")
        logger.info("Tip: Use 'init' command to create a new knowledge base")
    else:
        for kb_name in kb_list:
            default_marker = " ★(default)" if kb_name == default_kb else ""
            logger.info("  • %s%s", kb_name, default_marker)

            # Display statistics
            try:
                info = manager.get_info(kb_name)
                stats = info.get("statistics", {})
                logger.info("    - Documents: %d files", stats.get("raw_documents", 0))
                logger.info("    - Images: %d files", stats.get("images", 0))
                logger.info(
                    "    - RAG: %s",
                    "Initialized" if stats.get("rag_initialized") else "Not initialized",
                )
            except Exception:
                pass

    print("=" * 60 + "\n")


def show_kb_info(kb_name=None):
    """Display detailed knowledge base information"""
    manager = KnowledgeBaseManager(str(KNOWLEDGE_BASES_DIR))

    try:
        info = manager.get_info(kb_name)

        logger.section(f"Knowledge Base Info: {info['name']}")
        logger.info("Path: %s", info["path"])
        logger.info("Default: %s", "Yes" if info["is_default"] else "No")

        if info.get("metadata"):
            logger.info("[Metadata]")
            for key, value in info["metadata"].items():
                logger.info("  %s: %s", key, value)

        logger.info("[Statistics]")
        stats = info["statistics"]
        logger.info("  Raw Documents: %d files", stats["raw_documents"])
        logger.info("  Extracted Images: %d files", stats["images"])
        logger.info("  Content Lists: %d files", stats["content_lists"])
        logger.info(
            "  RAG Status: %s",
            "Initialized" if stats["rag_initialized"] else "Not initialized",
        )

        if "rag" in stats:
            print("\n[RAG Statistics]")
            for key, value in stats["rag"].items():
                print(f"  {key}: {value}")

        print("=" * 60 + "\n")

    except Exception as e:
        logger.error("Error: %s", e)


def set_default_kb(kb_name):
    """Set default knowledge base"""
    manager = KnowledgeBaseManager(str(KNOWLEDGE_BASES_DIR))

    try:
        manager.set_default(kb_name)
        logger.info("Set '%s' as default knowledge base", kb_name)
    except Exception as e:
        logger.error("Error: %s", e)


async def init_knowledge_base(args):
    """Initialize new knowledge base"""
    # Get API configuration
    env_config = get_env_config()
    api_key = args.api_key or env_config["api_key"]
    base_url = args.base_url or env_config["base_url"]

    if not api_key and not args.skip_processing:
        logger.error("API Key not set")
        logger.info("Please set environment variable LLM_API_KEY or use --api-key parameter")
        return

    # Collect document files
    # Use provider from env var or default to raganything (most comprehensive)
    import os

    provider = os.getenv("RAG_PROVIDER", "raganything")
    glob_patterns = FileTypeRouter.get_glob_patterns_for_provider(provider)

    doc_files = []
    if args.docs:
        doc_files.extend(args.docs)

    if args.docs_dir:
        docs_dir = Path(args.docs_dir)
        if docs_dir.exists() and docs_dir.is_dir():
            for pattern in glob_patterns:
                doc_files.extend([str(f) for f in docs_dir.glob(pattern)])
        else:
            logger.error("Document directory does not exist: %s", args.docs_dir)
            return

    if not args.skip_processing and not doc_files:
        logger.error("No documents specified")
        logger.info("Use --docs or --docs-dir to specify documents")
        return

    # Initialize knowledge base
    logger.section(f"Initializing knowledge base: {args.name}")

    initializer = KnowledgeBaseInitializer(
        kb_name=args.name,
        base_dir=str(KNOWLEDGE_BASES_DIR),
        api_key=api_key,
        base_url=base_url,
    )

    # Create directory structure
    initializer.create_directory_structure()

    # Copy documents
    if doc_files:
        copied_files = initializer.copy_documents(doc_files)
        logger.info("Copied %d files", len(copied_files))

    # Process documents
    if not args.skip_processing:
        await initializer.process_documents()
    else:
        logger.info("Skipping document processing")

    # Extract numbered items
    if not args.skip_processing and not args.skip_extract:
        initializer.extract_numbered_items(batch_size=args.batch_size)
    elif args.skip_extract:
        logger.info("Skipping numbered items extraction")

    logger.info("Knowledge base '%s' initialization complete!", args.name)
    logger.info("Location: %s", initializer.kb_dir)
    logger.info("=" * 60)


def extract_items(args):
    """Extract numbered items"""
    # Get API configuration
    env_config = get_env_config()
    api_key = args.api_key or env_config["api_key"]
    base_url = args.base_url or env_config["base_url"]

    if not api_key:
        print("✗ Error: API Key not set")
        print("Please set environment variable LLM_API_KEY or use --api-key parameter\n")
        return

    # Build paths
    kb_dir = KNOWLEDGE_BASES_DIR / args.kb
    content_list_dir = kb_dir / "content_list"

    if not content_list_dir.exists():
        logger.error("content_list directory does not exist: %s", content_list_dir)
        return

    # Get files to process
    if args.content_file:
        content_list_files = [content_list_dir / args.content_file]
        if not content_list_files[0].exists():
            logger.error("content_list file does not exist: %s", content_list_files[0])
            return
    else:
        content_list_files = sorted(content_list_dir.glob("*.json"))
        if not content_list_files:
            logger.error("No JSON files found in %s", content_list_dir)
            return

        if args.debug:
            logger.warning("Debug mode: Only processing first file")
            content_list_files = content_list_files[:1]

    output_file = kb_dir / "numbered_items.json"

    logger.section(f"Extracting numbered items: {args.kb}")
    logger.info("File count: %d", len(content_list_files))
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Max concurrent: %d", args.max_concurrent)
    logger.info("=" * 60)

    try:
        for idx, content_list_file in enumerate(content_list_files, 1):
            logger.info(
                "Processing file [%d/%d]: %s",
                idx,
                len(content_list_files),
                content_list_file.name,
            )

            process_content_list(
                content_list_file,
                output_file,
                api_key,
                base_url,
                args.batch_size,
                merge=(idx > 1),  # Auto-merge after first file
            )

        logger.info("Extraction complete!")
        logger.info("Output file: %s", output_file)

    except Exception as e:
        logger.error("Extraction failed: %s", e)


def delete_knowledge_base(args):
    """Delete knowledge base"""
    manager = KnowledgeBaseManager(str(KNOWLEDGE_BASES_DIR))

    try:
        success = manager.delete_knowledge_base(args.name, confirm=args.force)
        if success:
            logger.info("Deleted knowledge base '%s'", args.name)
    except Exception as e:
        logger.error("Error deleting knowledge base: %s", e)


def clean_rag_storage(args):
    """Clean RAG storage"""
    manager = KnowledgeBaseManager(str(KNOWLEDGE_BASES_DIR))

    logger.section("Cleaning RAG storage")

    try:
        manager.clean_rag_storage(args.name, backup=not args.no_backup)
        logger.info("RAG storage cleaned!")
        logger.info("Tip: Use 'add_documents.py' to reprocess documents to rebuild RAG")
        logger.info("%s", "=" * 60)
    except Exception as e:
        logger.error("Error cleaning RAG storage: %s", e)


async def refresh_knowledge_base(args):
    """Refresh knowledge base (reprocess all documents)"""
    manager = KnowledgeBaseManager(str(KNOWLEDGE_BASES_DIR))

    # Get API configuration
    env_config = get_env_config()
    api_key = args.api_key or env_config["api_key"]
    base_url = args.base_url or env_config["base_url"]

    if not api_key:
        print("✗ Error: API Key not set")
        print("Please set environment variable LLM_API_KEY or use --api-key parameter\n")
        return

    try:
        kb_name = args.name
        kb_dir = manager.get_knowledge_base_path(kb_name)
        raw_dir = kb_dir / "raw"

        if not raw_dir.exists() or not list(raw_dir.glob("*")):
            print(f"✗ Error: No raw documents found in knowledge base '{kb_name}'\n")
            return

        print("\n" + "=" * 60)
        print(f"🔄 Refreshing knowledge base: {kb_name}")
        print("=" * 60)
        print(f"Path: {kb_dir}")
        print("=" * 60 + "\n")

        # Step 1: Clean RAG storage
        print("Step 1/3: Cleaning RAG storage...")
        manager.clean_rag_storage(kb_name, backup=not args.no_backup)

        # Step 2: Clean content_list and images (optional)
        if args.full:
            print("\nStep 2/3: Cleaning extracted content and images...")
            content_list_dir = kb_dir / "content_list"
            images_dir = kb_dir / "images"

            if content_list_dir.exists():
                import shutil

                shutil.rmtree(content_list_dir)
                content_list_dir.mkdir(parents=True, exist_ok=True)
                print("  ✓ Cleaned content_list")

            if images_dir.exists():
                import shutil

                shutil.rmtree(images_dir)
                images_dir.mkdir(parents=True, exist_ok=True)
                print("  ✓ Cleaned images")
        else:
            print("\nStep 2/3: Skipping content cleanup (use --full for complete refresh)")

        # Step 3: Reprocess all documents
        print("\nStep 3/3: Reprocessing documents...")

        from src.knowledge.initializer import KnowledgeBaseInitializer

        initializer = KnowledgeBaseInitializer(
            kb_name=kb_name,
            base_dir=str(KNOWLEDGE_BASES_DIR),
            api_key=api_key,
            base_url=base_url,
        )

        # Reprocess documents
        await initializer.process_documents()

        # Extract numbered items
        if not args.skip_extract:
            print("\nExtracting numbered items...")
            initializer.extract_numbered_items(batch_size=args.batch_size)

        logger.info("Knowledge base '%s' refresh complete!", kb_name)
        logger.info("%s", "=" * 60)

    except Exception as e:
        logger.error("Refresh failed: %s", e)
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Knowledge Base Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:

  [Recommended: Directly run kb.py]
  python knowledge_init/kb.py list
  python knowledge_init/kb.py info ai_textbook
  python knowledge_init/kb.py set-default math2211
  python knowledge_init/kb.py init my_kb --docs document.pdf
  python knowledge_init/kb.py init my_course --docs-dir ./materials/
  python knowledge_init/kb.py extract --kb ai_textbook
  python knowledge_init/kb.py extract --kb ai_textbook --debug

  [New: Delete and Refresh Features]
  python knowledge_init/kb.py delete old_kb             # Delete knowledge base (requires confirmation)
  python knowledge_init/kb.py delete old_kb --force     # Force delete (skip confirmation)
  python knowledge_init/kb.py clean-rag C2-test         # Clean RAG storage (fix corrupted graph data)
  python knowledge_init/kb.py refresh ai_textbook       # Refresh knowledge base (reprocess all documents)
  python knowledge_init/kb.py refresh ai_textbook --full # Full refresh

  [Method 2: Run as module]
  python -m knowledge_init.start_kb list
  python -m knowledge_init.start_kb init my_kb --docs document.pdf
  python -m knowledge_init.start_kb clean-rag C2-test

  [Important] All commands must be run from project root directory (DeepTutor/)!
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # list command
    subparsers.add_parser("list", help="List all knowledge bases")

    # info command
    info_parser = subparsers.add_parser("info", help="Show knowledge base information")
    info_parser.add_argument(
        "name",
        nargs="?",
        help="Knowledge base name (optional, default shows default knowledge base)",
    )

    # set-default command
    default_parser = subparsers.add_parser("set-default", help="Set default knowledge base")
    default_parser.add_argument("name", help="Knowledge base name")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize new knowledge base")
    init_parser.add_argument("name", help="Knowledge base name")
    init_parser.add_argument("--docs", nargs="+", help="Document file list")
    init_parser.add_argument("--docs-dir", help="Document directory")
    init_parser.add_argument("--api-key", help="OpenAI API Key")
    init_parser.add_argument("--base-url", help="API Base URL")
    init_parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip document processing",
    )
    init_parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip numbered items extraction",
    )
    init_parser.add_argument("--batch-size", type=int, default=20, help="Batch size (default 20)")

    # extract command
    extract_parser = subparsers.add_parser("extract", help="Extract numbered items")
    extract_parser.add_argument("--kb", required=True, help="Knowledge base name")
    extract_parser.add_argument("--content-file", help="Specify content_list file (optional)")
    extract_parser.add_argument(
        "--batch-size", type=int, default=20, help="Batch size (default 20)"
    )
    extract_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent tasks (default 5)",
    )
    extract_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (only process first file)",
    )
    extract_parser.add_argument("--api-key", help="OpenAI API Key")
    extract_parser.add_argument("--base-url", help="API Base URL")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete knowledge base")
    delete_parser.add_argument("name", help="Knowledge base name")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation (dangerous)")

    # clean-rag command
    clean_parser = subparsers.add_parser(
        "clean-rag", help="Clean RAG storage (fix corrupted graph data)"
    )
    clean_parser.add_argument(
        "name",
        nargs="?",
        help="Knowledge base name (optional, default uses default knowledge base)",
    )
    clean_parser.add_argument(
        "--no-backup", action="store_true", help="No backup (not recommended)"
    )

    # refresh command
    refresh_parser = subparsers.add_parser(
        "refresh", help="Refresh knowledge base (reprocess all documents)"
    )
    refresh_parser.add_argument("name", help="Knowledge base name")
    refresh_parser.add_argument(
        "--full",
        action="store_true",
        help="Full refresh (clean all extracted content)",
    )
    refresh_parser.add_argument(
        "--no-backup", action="store_true", help="No backup for RAG storage"
    )
    refresh_parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip numbered items extraction",
    )
    refresh_parser.add_argument(
        "--batch-size", type=int, default=20, help="Batch size (default 20)"
    )
    refresh_parser.add_argument("--api-key", help="OpenAI API Key")
    refresh_parser.add_argument("--base-url", help="API Base URL")

    args = parser.parse_args()

    # Windows console UTF-8 support
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # Execute command
    if args.command == "list":
        list_knowledge_bases()

    elif args.command == "info":
        show_kb_info(args.name)

    elif args.command == "set-default":
        set_default_kb(args.name)

    elif args.command == "init":
        try:
            asyncio.run(init_knowledge_base(args))
        except (KeyboardInterrupt, SystemExit):
            print("\n\n⚠️  Operation cancelled")
        except IndexError as e:
            # Ignore IndexError during asyncio cleanup (doesn't affect functionality)
            if "pop from an empty deque" not in str(e):
                raise
        except Exception as e:
            print(f"\n✗ Error: {e}")
            raise

    elif args.command == "extract":
        extract_items(args)

    elif args.command == "delete":
        delete_knowledge_base(args)

    elif args.command == "clean-rag":
        clean_rag_storage(args)

    elif args.command == "refresh":
        try:
            asyncio.run(refresh_knowledge_base(args))
        except (KeyboardInterrupt, SystemExit):
            print("\n\n⚠️  Operation cancelled")
        except IndexError as e:
            # Ignore IndexError during asyncio cleanup
            if "pop from an empty deque" not in str(e):
                raise
        except Exception as e:
            print(f"\n✗ Error: {e}")
            raise

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
