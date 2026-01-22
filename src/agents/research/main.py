#!/usr/bin/env python
"""
DR-in-KG 2.0 - Main Entry
Deep research system based on dynamic topic queue

Usage:
  python main.py --topic "Research Topic" [--preset quick/standard/deep]
"""

import argparse
import asyncio
from pathlib import Path
import sys

from dotenv import load_dotenv
import yaml

from src.agents.research.research_pipeline import ResearchPipeline
from src.logging import get_logger
from src.services.llm import get_llm_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger = get_logger("ResearchCLI", console_output=True, file_output=True)


def load_config(
    config_path: str | None = None,
    preset: str | None = None,
) -> dict[str, object]:
    """Load the research configuration.

    Args:
        config_path: Optional configuration file path.
        preset: Preset mode (quick/standard/deep).
    Returns:
        The merged configuration dictionary.
    Raises:
        FileNotFoundError: When a custom config path does not exist.
    """
    if config_path is None:
        from src.services.config import load_config_with_main

        config = load_config_with_main("research_config.yaml", PROJECT_ROOT)
    else:
        # If custom config path provided, load it directly (for backward compatibility)
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Apply preset
    if preset and "presets" in config and preset in config["presets"]:
        logger.success(f"Applied preset configuration: {preset}")
        preset_config = config["presets"][preset]
        for key, value in preset_config.items():
            if key in config and isinstance(value, dict):
                config[key].update(value)

    return config


def display_config(config: dict[str, object]) -> None:
    """Display the current configuration.

    Args:
        config: The configuration dictionary.
    Returns:
        None.
    Raises:
        None.
    """
    logger.section("Current Configuration")

    planning = config.get("planning", {})
    researching = config.get("researching", {})
    reporting = config.get("reporting", {})

    if not isinstance(planning, dict):
        planning = {}
    if not isinstance(researching, dict):
        researching = {}
    if not isinstance(reporting, dict):
        reporting = {}

    decompose = planning.get("decompose", {})
    if not isinstance(decompose, dict):
        decompose = {}

    logger.info("Planning Configuration")
    logger.info(f"  Initial subtopics: {decompose.get('initial_subtopics', 5)}")
    logger.info(f"  Max subtopics: {decompose.get('max_subtopics', 10)}")

    logger.info("Researching Configuration")
    logger.info(f"  Max iterations: {researching.get('max_iterations', 5)}")
    logger.info(f"  Research mode: {researching.get('research_mode', 'deep')}")
    logger.info("  Enabled tools:")
    logger.info(f"    - RAG: {researching.get('enable_rag_hybrid', True)}")
    logger.info(f"    - Web Search: {researching.get('enable_web_search', True)}")
    logger.info(f"    - Paper Search: {researching.get('enable_paper_search', True)}")

    logger.info("Reporting Configuration")
    logger.info(
        f"  Min section length: {reporting.get('min_section_length', 500)} characters"
    )
    logger.info(
        f"  Enable topic deduplication: {reporting.get('enable_deduplication', True)}"
    )


async def main() -> None:
    """Run the research CLI entrypoint.

    Args:
        None.
    Returns:
        None.
    Raises:
        SystemExit: When configuration or execution fails.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="DR-in-KG 2.0 - Deep research system based on dynamic topic queue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick mode
  python main.py --topic "Deep Learning Basics" --preset quick

  # Standard mode
  python main.py --topic "Transformer Architecture" --preset standard

  # Deep mode
  python main.py --topic "Graph Neural Networks" --preset deep
        """,
    )

    parser.add_argument("--topic", type=str, required=True, help="Research topic")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Configuration file path (default: config/research_config.yaml)"),
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "standard", "deep"],
        help="Preset configuration (quick: fast, standard: standard, deep: deep)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config file)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check API configuration
    try:
        llm_config = get_llm_config()
    except ValueError as e:
        logger.error(f"LLM configuration error: {e}")
        logger.info("Please configure in .env or DeepTutor.env file:")
        logger.info("  LLM_MODEL=gpt-4o")
        logger.info("  LLM_API_KEY=your_api_key_here")
        logger.info("  LLM_HOST=https://api.openai.com/v1")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(args.config, args.preset)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e!s}")
        sys.exit(1)

    # Override configuration (command line arguments take priority)
    if args.output_dir:
        system_config = config.setdefault("system", {})
        if isinstance(system_config, dict):
            system_config["output_base_dir"] = args.output_dir
            system_config["reports_dir"] = args.output_dir

    # Display configuration
    display_config(config)

    # Create research pipeline
    pipeline = ResearchPipeline(
        config=config, api_key=llm_config.api_key, base_url=llm_config.base_url
    )

    # Execute research
    try:
        result = await pipeline.run(topic=args.topic)

        logger.section("Research completed")
        logger.info(f"Research ID: {result['research_id']}")
        logger.info(f"Topic: {result['topic']}")
        logger.info(f"Final Report: {result['final_report_path']}")

    except KeyboardInterrupt:
        logger.warning("Research interrupted by user")
        sys.exit(0)
    except Exception:
        logger.exception("Research failed")
        sys.exit(1)


if __name__ == "__main__":
    # Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
