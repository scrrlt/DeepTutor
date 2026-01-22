#!/usr/bin/env python
"""
Run Code Tool - Code execution tool
Execute Python code in isolated workspace, preserving original input/output structure.
"""

import ast
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_CODE_WORKSPACE_ENV = "RUN_CODE_WORKSPACE"
RUN_CODE_ALLOWED_ROOTS_ENV = "RUN_CODE_ALLOWED_ROOTS"
DEFAULT_WORKSPACE_NAME = "run_code_workspace"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.logging import get_logger

logger = get_logger("CodeExecutor")


def _load_config() -> dict[str, Any]:
    """Load run_code configuration from main.yaml and module configs"""
    try:
        from src.services.config import load_config_with_main

        # Try loading from solve_config (most common use case)
        try:
            config = load_config_with_main("solve_config.yaml", PROJECT_ROOT)
            run_code_config = config.get("tools", {}).get("run_code", {})
            if run_code_config:
                logger.debug("Loaded run_code config from solve_config.yaml (with main.yaml)")
                return run_code_config
        except Exception as e:
            logger.debug(f"Failed to load from solve_config: {e}")

        # Fallback to question_config
        try:
            config = load_config_with_main("question_config.yaml", PROJECT_ROOT)
            run_code_config = config.get("tools", {}).get("run_code", {})
            if run_code_config:
                logger.debug("Loaded run_code config from question_config.yaml (with main.yaml)")
                return run_code_config
        except Exception as e:
            logger.debug(f"Failed to load from question_config: {e}")

        # Fallback to main.yaml only
        try:
            config = load_config_with_main("solve_config.yaml", PROJECT_ROOT)
            run_code_config = config.get("tools", {}).get("run_code", {})
            if run_code_config:
                return run_code_config
        except Exception as e:
            logger.debug("Failed to load from solve_config (fallback): %s", e)

    except ImportError:
        logger.debug("config_loader not available, using fallback")

    # Fallback: try loading main.yaml directly
    try:
        import yaml

        main_config_path = PROJECT_ROOT / "config" / "main.yaml"
        if main_config_path.exists():
            with open(main_config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            run_code_config = config.get("tools", {}).get("run_code", {})
            if run_code_config:
                logger.debug("Loaded run_code config from main.yaml")
                return run_code_config
    except Exception as e:
        logger.debug(f"Failed to load from main.yaml: {e}")

    return {}


class CodeExecutionError(Exception):
    """Code execution error"""


@dataclass
class OperationEntry:
    action: str
    details: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class OperationLogger:
    """Simple operation history logger, inspired by code_implementation_server recording method"""

    def __init__(self, max_entries: int = 200):
        self._history: list[OperationEntry] = []
        self._max_entries = max_entries

    def log(self, action: str, details: dict[str, Any]):
        entry = OperationEntry(action=action, details=details)
        self._history.append(entry)
        if len(self._history) > self._max_entries:
            self._history.pop(0)
        logger.debug(f"Operation logged: {action} | details={details.get('status')}")

    @property
    def history(self) -> list[OperationEntry]:
        return list(self._history)


from .code_runner.workspace import WorkspaceManager

# Backwards-compatible alias for legacy imports
# (older code may import WorkspaceManager from this module)


class ImportGuard:
    """Parse AST, restrict import modules, ensure consistency with allowed_imports logic"""

    @staticmethod
    def validate(code: str, allowed_imports: list[str] | None):
        if not allowed_imports:
            return

        allowed = set(allowed_imports)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise CodeExecutionError(f"Code syntax error: {exc}") from exc

        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.append(node.module.split(".")[0])

        unauthorized = sorted({name for name in imported if name not in allowed})
        if unauthorized:
            raise CodeExecutionError(
                f"The following modules are not in the allowed list: {', '.join(unauthorized)}"
            )


WORKSPACE_MANAGER = WorkspaceManager()
OPERATION_LOGGER = OperationLogger()
# Note: The original synchronous CodeExecutionEnvironment has been removed
# in favor of the asyncio-based LocalProcessRunner in src.tools.code_runner.
# The run_code entry point delegates to the runner abstraction.


async def run_code(
    language: str,
    code: str,
    timeout: int = 10,
    assets_dir: str | None = None,
    allowed_imports: list[str] | None = None,
) -> dict[str, Any]:
    """
    Execute code in isolated environment, return result structure consistent with previous version.
    """
    if language.lower() != "python":
        raise ValueError(f"Unsupported language: {language}, currently only Python is supported")

    WORKSPACE_MANAGER.ensure_initialized()
    ImportGuard.validate(code, allowed_imports)

    assets_path = WORKSPACE_MANAGER.resolve_assets_dir(assets_dir)

    # Prepare execution options and delegate to Runner abstraction
    from .code_runner.runner import LocalProcessRunner
    from .code_runner.types import ExecutionOptions

    options = ExecutionOptions(
        timeout=timeout,
        assets_dir=assets_path,
        allowed_imports=allowed_imports,
    )
    runner = LocalProcessRunner(workspace_manager=WORKSPACE_MANAGER)

    run_result = await runner.run(code, options)

    # Convert RunResult into legacy-compatible dict
    artifacts, artifact_paths = WORKSPACE_MANAGER.collect_artifacts(assets_path)

    OPERATION_LOGGER.log(
        "execute_python",
        {
            "status": "success" if run_result.exit_code == 0 else "error",
            "language": language,
            "timeout": timeout,
            "assets_dir": str(assets_path) if assets_path else None,
            "exit_code": run_result.exit_code,
            "elapsed_ms": run_result.elapsed_ms,
            "code_size": len(code),
        },
    )

    return {
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
        "artifacts": artifacts,
        "artifact_paths": artifact_paths,
        "exit_code": run_result.exit_code,
        "elapsed_ms": run_result.elapsed_ms,
    }


def run_code_sync(
    language: str,
    code: str,
    timeout: int = 10,
    assets_dir: str | None = None,
) -> dict[str, Any]:
    """
    Synchronous version of code execution (for non-async environments)
    """

    return asyncio.run(run_code(language, code, timeout, assets_dir))


if __name__ == "__main__":
    import textwrap

    async def _demo():
        logger.info("==== 1. Test normal output ====")
        sample1 = "print('Hello from run_code workspace!')"
        result1 = await run_code("python", sample1, timeout=5)
        logger.info("stdout: %s", result1["stdout"])
        logger.info("stderr: %s", result1["stderr"])
        logger.info("artifacts: %s", result1.get("artifacts", {}))
        logger.info("artifact_paths: %s", result1.get("artifact_paths", []))
        logger.info("exit_code: %s", result1["exit_code"])
        logger.info("-" * 40)

        logger.info("==== 2. Test exception case ====")
        sample2 = "raise ValueError('Test error from run_code!')"
        result2 = await run_code("python", sample2, timeout=5)
        logger.info("stdout: %s", result2["stdout"])
        logger.info("stderr: %s", result2["stderr"])
        logger.info("exit_code: %s", result2["exit_code"])
        logger.info("-" * 40)

        logger.info("==== 3. Test code timeout ====")
        sample3 = textwrap.dedent(
            """
        import time
        time.sleep(10)
        print("Timeout should occur before this prints.")
        """
        )
        result3 = await run_code("python", sample3, timeout=2)
        logger.info("stdout: %s", result3["stdout"])
        logger.info("stderr: %s", result3["stderr"])
        logger.info("exit_code: %s", result3["exit_code"])
        logger.info("-" * 40)

        logger.info("==== 4. Test plotting functionality (matplotlib) ====")
        sample4 = textwrap.dedent(
            """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([1, 2, 3], [4, 2, 5])
        plt.title('Simple Plot')
        plt.savefig('test_plot.png')
        print('Plot created!')
        """
        )
        result4 = await run_code("python", sample4, timeout=5)
        logger.info("stdout: %s", result4["stdout"])
        logger.info("stderr: %s", result4["stderr"])
        logger.info("artifacts: %s", result4.get("artifacts", {}))
        logger.info("artifact_paths: %s", result4.get("artifact_paths", []))
        logger.info("exit_code: %s", result4["exit_code"])
        # Check generated images
        if result4.get("artifact_paths"):
            logger.info("Generated image files: %s", result4["artifact_paths"])
        else:
            logger.info("No image files found.")
        logger.info("-" * 40)

        logger.info("==== 5. Test standard input ====")
        sample5 = textwrap.dedent(
            """
        text = input("Please enter content: ")
        print("You entered: ", text)
        """
        )
        # Standard run_code does not provide stdin, this example tests output behavior
        result5 = await run_code("python", sample5, timeout=5)
        logger.info("stdout: %s", result5["stdout"])
        logger.info("stderr: %s", result5["stderr"])
        logger.info("exit_code: %s", result5["exit_code"])
        logger.info("-" * 40)

        logger.info("==== 6. Test multi-file and resource read/write ====")
        sample6 = textwrap.dedent(
            """
        with open('test_file.txt', 'w', encoding='utf-8') as f:
            f.write('Fake data for test!\\nAnother line.')
        with open('test_file.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        print('File content:', content)
        """
        )
        result6 = await run_code("python", sample6, timeout=5)
        logger.info("stdout: %s", result6["stdout"])
        logger.info("stderr: %s", result6["stderr"])
        logger.info("artifacts: %s", result6.get("artifacts", {}))
        logger.info("artifact_paths: %s", result6.get("artifact_paths", []))
        logger.info("exit_code: %s", result6["exit_code"])
        logger.info("%s", "-" * 40)

    asyncio.run(_demo())
