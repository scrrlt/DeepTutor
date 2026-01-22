#!/usr/bin/env python
"""DeepTutor CLI launcher.

Provides a command-line interface for the solver, question generation, research,
idea generation, and web service workflows.

Args:
    None.
Returns:
    None.
Raises:
    None.
"""

import asyncio
from pathlib import Path
import sys

from dotenv import load_dotenv

# Set Windows console UTF-8 encoding
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Load environment variables
load_dotenv()

# Add project root directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agents.question import AgentCoordinator
from src.agents.solve import MainSolver
from src.api.utils.history import ActivityType, history_manager
from src.logging import get_logger
from src.services.llm import get_llm_config

# Initialize logger for CLI
logger = get_logger("CLI", console_output=True, file_output=True)


class AITutorStarter:
    """DeepTutor CLI launcher.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    def __init__(self) -> None:
        """Initialize the CLI launcher.

        Args:
            None.
        Returns:
            None.
        Raises:
            SystemExit: Raised when LLM configuration is missing.
        """
        # Initialize user data directories
        try:
            from src.services.setup import init_user_directories

            init_user_directories(project_root)
        except Exception as e:
            logger.warning(f"Failed to initialize user directories: {e}")
            logger.info("Continuing anyway...")

        try:
            llm_config = get_llm_config()
            self.api_key: str = llm_config.api_key
            self.base_url: str | None = llm_config.base_url
        except ValueError as e:
            logger.error(str(e))
            logger.error("Please configure LLM settings in .env or DeepTutor.env file")
            sys.exit(1)

        # Load knowledge base list
        self.available_kbs = self._load_available_kbs()

        logger.section("DeepTutor Intelligent Teaching Assistant System")
        logger.success("API configuration loaded")
        logger.info(f"Available knowledge bases: {', '.join(self.available_kbs)}")
        logger.separator()

    def _load_available_kbs(self) -> list[str]:
        """Load available knowledge base list.

        Args:
            None.
        Returns:
            The list of available knowledge base names.
        Raises:
            None.
        """
        kb_base_dir = project_root / "data" / "knowledge_bases"
        if not kb_base_dir.exists():
            return ["ai_textbook"]  # Default knowledge base

        # Read configuration file
        kb_config_file = kb_base_dir / "kb_config.json"
        if kb_config_file.exists():
            import json

            with open(kb_config_file, encoding="utf-8") as f:
                config = json.load(f)
                return list(config.get("knowledge_bases", {}).keys())

        # Otherwise scan directory
        kbs = []
        for item in kb_base_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                kbs.append(item.name)

        return kbs if kbs else ["ai_textbook"]

    def show_main_menu(self) -> str:
        """Display the main menu and get user selection.

        Args:
            None.
        Returns:
            The selected menu option.
        Raises:
            None.
        """
        logger.section("Please select a function:")
        logger.info(
            "1. 🧠 Solver System (Solve) - Intelligent academic problem solving"
        )
        logger.info("2. 📝 Question Generator (Question) - Generate questions")
        logger.info("3. 🔬 Deep Research (Research) - Multi-round research")
        logger.info("4. 💡 Idea Generation (IdeaGen) - Extract research ideas")
        logger.info("5. 🌐 Start Web Service (Web) - Start frontend/backend")
        logger.info("6. ⚙️  System Settings (Settings) - View configuration")
        logger.info("7. 🚪 Exit")
        logger.separator()

        while True:
            choice = input("\nPlease enter option (1-7): ").strip()
            if choice in ["1", "2", "3", "4", "5", "6", "7"]:
                return choice
            logger.warning("Invalid option, please try again")

    def select_kb(self, default: str | None = None) -> str:
        """Select a knowledge base.

        Args:
            default: Optional default knowledge base name.
        Returns:
            The selected knowledge base name.
        Raises:
            None.
        """
        if len(self.available_kbs) == 1:
            return self.available_kbs[0]

        logger.separator()
        logger.info("Available knowledge bases:")
        logger.separator()
        for i, kb in enumerate(self.available_kbs, 1):
            default_mark = " (default)" if kb == default else ""
            logger.info(f"{i}. {kb}{default_mark}")
        logger.separator()

        while True:
            choice = input(
                "\nPlease select knowledge base "
                f"(1-{len(self.available_kbs)}) [default: 1]: "
            ).strip()
            if not choice:
                return self.available_kbs[0]

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_kbs):
                    return self.available_kbs[idx]
                logger.warning(
                    f"Please enter a number between 1 and {len(self.available_kbs)}"
                )
            except ValueError:
                logger.warning("Please enter a valid number")

    def _prompt_text(self, prompt: str) -> str | None:
        """Prompt the user for a single-line response.

        Args:
            prompt: The prompt message to display.
        Returns:
            The response string, or None when empty.
        Raises:
            None.
        """
        logger.separator()
        logger.info(prompt)
        logger.separator()
        response = input().strip()
        return response or None

    def _prompt_multiline(self, prompt: str) -> str | None:
        """Prompt the user for a multi-line response.

        Args:
            prompt: The prompt message to display.
        Returns:
            The response string, or None when empty.
        Raises:
            None.
        """
        logger.separator()
        logger.info(prompt)
        logger.separator()

        lines: list[str] = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        if not lines:
            return None

        return "\n".join(lines)

    def _prompt_choice(
        self,
        title: str,
        prompt: str,
        options: list[tuple[str, str, str]],
        default_key: str,
    ) -> str:
        """Prompt the user to choose from options.

        Args:
            title: Section title to display before options.
            prompt: Input prompt string.
            options: List of tuples (key, label, value).
            default_key: Default option key when input is empty.
        Returns:
            The selected option value.
        Raises:
            None.
        """
        logger.separator()
        logger.info(title)
        logger.separator()
        value_map = {key: value for key, _, value in options}
        for key, label, _ in options:
            logger.info(f"{key}. {label}")
        logger.separator()

        while True:
            choice = input(prompt).strip()
            if not choice:
                return value_map[default_key]
            if choice in value_map:
                return value_map[choice]
            logger.warning("Invalid option, please try again")

    async def run_solve_mode(self) -> None:
        """Run solver mode.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        logger.section("Solver System")

        # Select knowledge base
        kb_name = self.select_kb(default="ai_textbook")
        logger.success(f"Selected knowledge base: {kb_name}")

        question = self._prompt_multiline(
            "Please enter your question (multi-line input supported, "
            "empty line to finish):"
        )
        if not question:
            logger.warning("No question entered, returning to main menu")
            return

        # Display solver mode description
        logger.separator()
        logger.info("Solver Mode: Dual-Loop Architecture")
        logger.separator()
        logger.info("Analysis Loop: Deep understanding of user question")
        logger.info("   Investigate → Note")
        logger.info("")
        logger.info("Solve Loop: Collaborative solving")
        logger.info("   Manager → Solve → Tool → Response → Precision")
        logger.separator()

        logger.section("Starting solver...")

        try:
            # Create solver
            solver = MainSolver(
                config_path=None,  # Use default configuration file
                api_key=self.api_key,
                base_url=self.base_url,
                kb_name=kb_name,
            )

            # Run solver
            result = await solver.solve(question=question, verbose=True)

            logger.section("Solving completed!")
            logger.info(f"Output directory: {result['metadata']['output_dir']}")

            logger.info("Solving statistics:")
            logger.info(f"   Execution mode: {result['metadata']['mode']}")
            logger.info(f"   Pipeline: {result.get('pipeline', 'dual_loop')}")

            if "analysis_iterations" in result:
                logger.info(
                    f"   Analysis loop iterations: {result['analysis_iterations']} rounds"
                )
            if "solve_steps" in result:
                logger.info(f"   Solve steps completed: {result['solve_steps']} steps")
            if "total_steps" in result:
                logger.info(f"   Total planned steps: {result['total_steps']}")
            if "citations" in result:
                logger.info(f"   Citation count: {len(result['citations'])}")

            logger.info("Output files:")
            logger.info(f"   Markdown: {result['output_md']}")
            logger.info(f"   JSON: {result['output_json']}")
            logger.separator()

            # Display answer preview
            formatted_solution = result.get("formatted_solution", "")
            if formatted_solution:
                logger.separator()
                logger.info("Answer preview:")
                logger.separator()
                preview_lines = formatted_solution.split("\n")[:20]
                preview = "\n".join(preview_lines)
                if len(formatted_solution) > len(preview):
                    preview += "\n\n... (more content available in output file) ..."
                logger.info(preview)
                logger.separator()

            # Save to history
            try:
                history_manager.add_entry(
                    activity_type=ActivityType.SOLVE,
                    title=question[:50] + "..." if len(question) > 50 else question,
                    content={
                        "question": question,
                        "answer": result.get("formatted_solution", ""),
                        "output_dir": result["metadata"]["output_dir"],
                        "kb_name": kb_name,
                        "metadata": result.get("metadata", {}),
                    },
                    summary=(
                        formatted_solution[:100] + "..."
                        if formatted_solution and len(formatted_solution) > 100
                        else (formatted_solution or "")
                    ),
                )
            except Exception as hist_error:
                # History save failure does not affect main flow
                logger.warning(f"History save failed: {hist_error!s}")

        except Exception as e:
            logger.section("Solving failed")
            logger.error(str(e))
            import traceback

            logger.debug("Debug information:")
            logger.debug(traceback.format_exc())

    async def run_question_mode(self) -> None:
        """Run question generation mode.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        logger.section("Question Generation System")

        # Select knowledge base
        kb_name = self.select_kb(default="ai_textbook")
        logger.success(f"Selected knowledge base: {kb_name}")

        knowledge_point = self._prompt_text("Please enter knowledge point:")
        if not knowledge_point:
            logger.warning("No knowledge point entered, returning to main menu")
            return

        difficulty = self._prompt_choice(
            title="Difficulty selection:",
            prompt="\nPlease select difficulty (1-3) [default: 2]: ",
            options=[
                ("1", "Easy", "easy"),
                ("2", "Medium", "medium"),
                ("3", "Hard", "hard"),
            ],
            default_key="2",
        )

        question_type = self._prompt_choice(
            title="Question type selection:",
            prompt="\nPlease select question type (1/2) [default: 1]: ",
            options=[
                ("1", "Multiple choice (choice)", "choice"),
                ("2", "Written answer (written)", "written"),
            ],
            default_key="1",
        )

        logger.section("Starting question generation")

        try:
            # Create coordinator
            coordinator = AgentCoordinator(
                api_key=self.api_key,
                base_url=self.base_url,
                kb_name=kb_name,
                output_dir="./user/question",
            )

            # Build requirement
            requirement = {
                "knowledge_point": knowledge_point,
                "difficulty": difficulty,
                "question_type": question_type,
                "additional_requirements": "Ensure questions are clear and academically rigorous",
            }

            # Run question generation
            result = await coordinator.generate_question(requirement)

            if result.get("success"):
                logger.section("Question generation completed")

                question_data = result.get("question", {})
                logger.info("Question:")
                logger.info(question_data.get("question", ""))

                if question_data.get("options"):
                    logger.info("Options:")
                    for key, value in question_data.get("options", {}).items():
                        logger.info(f"  {key}. {value}")

                logger.info(f"Answer: {question_data.get('correct_answer', '')}")
                logger.info("Explanation:")
                logger.info(question_data.get("explanation", ""))

                if result.get("output_dir"):
                    logger.info(f"Output directory: {result['output_dir']}")

                # Save to history
                try:
                    history_manager.add_entry(
                        activity_type=ActivityType.QUESTION,
                        title=f"{knowledge_point} ({question_type})",
                        content={
                            "requirement": requirement,
                            "question": question_data,
                            "output_dir": result.get("output_dir", ""),
                            "kb_name": kb_name,
                        },
                        summary=(
                            question_data.get("question", "")[:100] + "..."
                            if len(question_data.get("question", "")) > 100
                            else question_data.get("question", "")
                        ),
                    )
                except Exception as hist_error:
                    # History save failure does not affect main flow
                    logger.warning(f"History save failed: {hist_error!s}")
            else:
                logger.section("Question generation failed")
                logger.error(result.get("error", "Unknown error"))
                if result.get("reason"):
                    logger.error(f"Reason: {result['reason']}")

        except Exception as e:
            logger.section("Question generation failed")
            logger.error(str(e))
            import traceback

            logger.debug("Debug information:")
            logger.debug(traceback.format_exc())

    async def run_research_mode(self) -> None:
        """Run deep research mode.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        logger.section("Deep Research System")

        # Select knowledge base
        kb_name = self.select_kb(default="ai_textbook")
        logger.success(f"Selected knowledge base: {kb_name}")

        topic = self._prompt_text("Please enter research topic:")
        if not topic:
            logger.warning("No topic entered, returning to main menu")
            return

        preset = self._prompt_choice(
            title="Research mode:",
            prompt="\nPlease select mode (1-4) [default: 1]: ",
            options=[
                ("1", "Quick - 2 subtopics, 2 iterations", "quick"),
                ("2", "Standard - 5 subtopics, 5 iterations", "standard"),
                ("3", "Deep - 8 subtopics, 7 iterations", "deep"),
                ("4", "Auto - Automatically generate subtopic count", "auto"),
            ],
            default_key="1",
        )

        logger.section("Starting deep research")

        try:
            # Import research pipeline
            from src.agents.research.research_pipeline import ResearchPipeline
            from src.services.config import load_config_with_main

            # Load configuration using unified config loader
            config = load_config_with_main("main.yaml", project_root)

            # Extract research.* configs to top level (ResearchPipeline expects flat structure)
            research_config = config.get("research", {})
            if "planning" not in config:
                config["planning"] = research_config.get("planning", {}).copy()
            if "researching" not in config:
                config["researching"] = research_config.get("researching", {}).copy()
            if "reporting" not in config:
                config["reporting"] = research_config.get("reporting", {}).copy()
            if "rag" not in config:
                config["rag"] = research_config.get("rag", {}).copy()
            if "queue" not in config:
                config["queue"] = research_config.get("queue", {}).copy()
            if "presets" not in config:
                config["presets"] = research_config.get("presets", {}).copy()

            # Set output paths
            if "system" not in config:
                config["system"] = {}
            output_base = project_root / "data" / "user" / "research"
            config["system"]["output_base_dir"] = str(output_base / "cache")
            config["system"]["reports_dir"] = str(output_base / "reports")

            # Apply preset mode configuration
            preset_configs = {
                "quick": {
                    "planning": {
                        "decompose": {"initial_subtopics": 2, "mode": "manual"}
                    },
                    "researching": {"max_iterations": 2},
                },
                "standard": {
                    "planning": {
                        "decompose": {"initial_subtopics": 5, "mode": "manual"}
                    },
                    "researching": {"max_iterations": 5},
                },
                "deep": {
                    "planning": {
                        "decompose": {"initial_subtopics": 8, "mode": "manual"}
                    },
                    "researching": {"max_iterations": 7},
                },
                "auto": {
                    "planning": {
                        "decompose": {"mode": "auto", "auto_max_subtopics": 8}
                    },
                    "researching": {"max_iterations": 6},
                },
            }

            if preset in preset_configs:
                preset_cfg = preset_configs[preset]
                # Apply planning configuration
                if "planning" in preset_cfg:
                    for key, value in preset_cfg["planning"].items():
                        if key not in config["planning"]:
                            config["planning"][key] = {}
                        config["planning"][key].update(value)
                # Apply researching configuration
                if "researching" in preset_cfg:
                    config["researching"].update(preset_cfg["researching"])

                if preset == "auto":
                    logger.success(
                        "Auto mode enabled (automatically generate subtopic "
                        "count, max: "
                        f"{config['planning']['decompose'].get('auto_max_subtopics', 8)})"
                    )
                else:
                    logger.success(f"{preset.capitalize()} mode enabled")

            # Create research pipeline
            pipeline = ResearchPipeline(
                config=config,
                api_key=self.api_key,
                base_url=self.base_url,
                kb_name=kb_name,
            )

            # Execute research
            result = await pipeline.run(topic)

            logger.section("Research completed")
            logger.info(f"Report location: {result['final_report_path']}")
            logger.info(f"Research ID: {result['research_id']}")

            metadata = result.get("metadata", {})
            if metadata:
                logger.info("Research statistics:")
                logger.info(
                    f"   Report word count: {metadata.get('report_word_count', 0)}"
                )
                stats = metadata.get("statistics", {})
                if stats:
                    logger.info(f"   Topic blocks: {stats.get('total_blocks', 0)}")
                    logger.info(f"   Completed topics: {stats.get('completed', 0)}")
                    logger.info(f"   Tool calls: {stats.get('total_tool_calls', 0)}")

            # Save to history
            try:
                # Read report content
                report_content = ""
                if (
                    result.get("final_report_path")
                    and Path(result["final_report_path"]).exists()
                ):
                    with open(result["final_report_path"], encoding="utf-8") as f:
                        report_content = f.read()

                history_manager.add_entry(
                    activity_type=ActivityType.RESEARCH,
                    title=topic,
                    content={
                        "topic": topic,
                        "report": report_content,
                        "report_path": result.get("final_report_path", ""),
                        "research_id": result.get("research_id", ""),
                        "kb_name": kb_name,
                        "metadata": metadata,
                    },
                    summary=(
                        report_content[:200] + "..."
                        if len(report_content) > 200
                        else report_content
                    ),
                )
            except Exception as hist_error:
                # History save failure does not affect main flow
                logger.warning(f"History save failed: {hist_error!s}")

        except Exception as e:
            logger.section("Research failed")
            logger.error(str(e))
            import traceback

            logger.debug("Debug information:")
            logger.debug(traceback.format_exc())

    async def run_ideagen_mode(self) -> None:
        """Run idea generation mode.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        logger.section("Idea Generation System")

        # Select knowledge base
        kb_name = self.select_kb()
        logger.success(f"Selected knowledge base: {kb_name}")

        materials = self._prompt_multiline(
            "Please enter knowledge points or material content "
            "(multi-line input supported, empty line to finish):"
        )
        if not materials:
            logger.warning("No content entered, returning to main menu")
            return

        logger.section("Starting research idea generation")

        try:
            # Import ideagen module
            from src.agents.ideagen.idea_generation_workflow import (
                IdeaGenerationWorkflow,
            )
            from src.agents.ideagen.material_organizer_agent import (
                MaterialOrganizerAgent,
            )

            # Organize materials
            organizer = MaterialOrganizerAgent(
                api_key=self.api_key, base_url=self.base_url
            )

            logger.info("Extracting knowledge points...")
            knowledge_points = await organizer.extract_knowledge_points(materials)
            logger.success(f"Extracted {len(knowledge_points)} knowledge points")

            if not knowledge_points:
                logger.warning("Failed to extract valid knowledge points")
                return

            # Generate ideas
            workflow = IdeaGenerationWorkflow(
                api_key=self.api_key, base_url=self.base_url
            )

            logger.info("Generating research ideas...")
            result = await workflow.process(knowledge_points)
            logger.section("Research idea generation completed")
            logger.info(result)

        except Exception as e:
            logger.section("Generation failed")
            logger.error(str(e))
            import traceback

            logger.debug("Debug information:")
            logger.debug(traceback.format_exc())

    def run_web_mode(self) -> None:
        """Start the web service.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        logger.section("Start Web Service")
        logger.info("1. Start backend API only (port 8000)")
        logger.info("2. Start frontend only (port 3000)")
        logger.info("3. Start both frontend and backend")
        logger.info("4. Return to main menu")

        while True:
            choice = input("\nPlease select (1-4): ").strip()
            if choice == "1":
                logger.section("Starting backend service")
                logger.info(
                    "Command: python -m uvicorn api.main:app "
                    "--host 127.0.0.1 --port 8000 --reload"
                )
                import subprocess  # nosec B404

                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "api.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8000",
                        "--reload",
                    ],
                    check=False,
                    shell=False,
                )
                break
            if choice == "2":
                logger.section("Starting frontend service")
                logger.info("Command: cd web && npm run dev")
                import subprocess  # nosec B404

                web_dir = project_root / "web"
                subprocess.run(["npm", "run", "dev"], check=False, cwd=web_dir)
                break
            if choice == "3":
                logger.section("Starting frontend and backend services")
                logger.info("Command: python start_web.py")
                import subprocess  # nosec B404

                subprocess.run([sys.executable, "start_web.py"], check=False)
                break
            if choice == "4":
                return
            logger.warning("Invalid option, please try again")

    def show_settings(self) -> None:
        """Display settings.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        logger.section("System Settings")

        # Display LLM configuration
        try:
            llm_config = get_llm_config()
            logger.info("LLM Configuration:")
            logger.info(f"   Model: {llm_config.model or 'N/A'}")
            logger.info(f"   API Endpoint: {llm_config.base_url or 'N/A'}")
            logger.info(
                "   API Key: "
                f"{'Configured' if llm_config.api_key else 'Not configured'}"
            )
        except Exception as e:
            logger.error(f"Load failed: {e}")

        # Display knowledge bases
        logger.info("Available knowledge bases:")
        for i, kb in enumerate(self.available_kbs, 1):
            logger.info(f"   {i}. {kb}")

        # Display configuration file locations
        logger.info("Configuration file locations:")
        env_files = [".env", "DeepTutor.env"]
        for env_file in env_files:
            env_path = project_root / env_file
            if env_path.exists():
                logger.info(f"   ✅ {env_path}")
            else:
                logger.info(f"   ⚪ {env_path} (not found)")

        logger.info(
            "Tip: To modify settings, edit the .env file directly, or use "
            "the Settings page in the Web interface."
        )

        input("\nPress Enter to return to main menu...")

    async def run(self) -> None:
        """Run the main loop.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        while True:
            try:
                choice = self.show_main_menu()

                if choice == "1":
                    await self.run_solve_mode()
                elif choice == "2":
                    await self.run_question_mode()
                elif choice == "3":
                    await self.run_research_mode()
                elif choice == "4":
                    await self.run_ideagen_mode()
                elif choice == "5":
                    self.run_web_mode()
                    continue  # Don't ask to continue after web mode
                elif choice == "6":
                    self.show_settings()
                    continue  # Don't ask to continue after settings
                elif choice == "7":
                    logger.section(
                        "Thank you for using DeepTutor Intelligent Teaching "
                        "Assistant System"
                    )
                    break

                # Ask if continue
                logger.separator()
                continue_choice = (
                    input("Continue using? (y/n) [default: y]: ").strip().lower()
                )
                if continue_choice == "n":
                    logger.section(
                        "Thank you for using DeepTutor Intelligent Teaching "
                        "Assistant System"
                    )
                    break

            except KeyboardInterrupt:
                logger.section("Program interrupted, thank you for using")
                break
            except Exception as e:
                logger.error(f"Error occurred: {e!s}")
                logger.info("Please retry or exit the program")


def main() -> None:
    """Run the CLI entrypoint.

    Args:
        None.
    Returns:
        None.
    Raises:
        SystemExit: Raised when startup fails.
    """
    try:
        starter = AITutorStarter()
        asyncio.run(starter.run())
    except Exception as e:
        logger.error(f"Startup failed: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    # Initialize user data directories
    try:
        from src.services.setup import init_user_directories

        init_user_directories(project_root)
    except Exception as e:
        logger.warning(f"Failed to initialize user directories: {e}")
        logger.info("Continuing anyway...")

    main()
