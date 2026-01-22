#!/usr/bin/env python
"""
DeepTutor One-Click Installation Script

Automatically installs all frontend and backend dependencies without interaction.
Execution flow: Install backend -> Install frontend -> Verify all packages
"""

import os
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logging import Logger, get_logger  # noqa: E402

logger: Logger = get_logger("Installer")

# Set Windows console UTF-8 encoding
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def log_step(message: str) -> None:
    """Log step information."""
    logger.info("\n" + "=" * 60)
    logger.info(f"📦 {message}")
    logger.info("=" * 60)


def log_success(message: str) -> None:
    """Log success message."""
    logger.info(f"✅ {message}")


def log_error(message: str) -> None:
    """Log error message."""
    logger.error(f"❌ {message}")


def log_info(message: str) -> None:
    """Log info message."""
    logger.info(f"ℹ️  {message}")


def log_warning(message: str) -> None:
    """Log warning message."""
    logger.warning(f"⚠️  {message}")


def check_virtual_env() -> None:
    """Check if running in a virtual environment (warning only)."""
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")

    if conda_env:
        log_success(f"Conda environment detected: {conda_env}")
    elif in_venv:
        log_success(f"Virtual environment detected: {sys.prefix}")
    else:
        log_warning("No isolated environment detected")
        log_info("Recommended: use conda or venv for isolation")
        log_info("Continuing installation...\n")


def install_with_uv(requirements_file: Path, project_root: Path) -> bool:
    """Try installing dependencies using uv (faster and better resolver)"""
    uv_path = shutil.which("uv")

    if not uv_path:
        # Try to install uv first
        log_info("uv not found, attempting to install uv...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "uv"],
                check=False,
                timeout=120,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                log_success("uv installed successfully")
            else:
                return False
        except Exception:
            return False

    log_info("Using uv for faster dependency resolution...")
    try:
        cmd = [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "-r",
            str(requirements_file),
        ]
        result = subprocess.run(
            cmd,
            check=False,
            cwd=project_root,
            timeout=900,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        log_warning(f"uv installation failed: {e}")
        return False


def install_with_pip_staged(requirements_file: Path, project_root: Path) -> bool:
    """Install dependencies in stages to avoid resolution-too-deep error"""
    log_info("Using staged pip installation to avoid dependency resolution issues...")

    # Stage 1: Install core dependencies first (without complex RAG packages)
    core_deps = [
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0",
        "tiktoken>=0.5.0",
        "requests>=2.31.0",
        "openai>=1.0.0",
        "aiohttp>=3.9.0",
        "httpx>=0.25.0",
        "nest_asyncio>=1.5.8",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.24.0",
        "websockets>=12.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "arxiv>=2.0.0",
        "pre-commit>=3.0.0",
    ]

    log_info("Stage 1/3: Installing core dependencies...")
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + core_deps
        result = subprocess.run(
            cmd,
            check=False,
            cwd=project_root,
            timeout=600,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            log_error("Failed to install core dependencies")
            return False
        log_success("Core dependencies installed")
    except Exception as e:
        log_error(f"Error installing core dependencies: {e}")
        return False

    # Stage 2: Install llama-index
    log_info("Stage 2/3: Installing llama-index...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "llama-index>=0.14.12"]
        result = subprocess.run(
            cmd,
            check=False,
            cwd=project_root,
            timeout=600,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            log_warning("llama-index installation had issues, continuing...")
    except Exception as e:
        log_warning(f"llama-index installation error: {e}")

    # Stage 3: Install raganything (includes lightrag-hku as dependency)
    log_info("Stage 3/4: Installing raganything (includes lightrag-hku, this may take a while)...")
    try:
        # First try normal install
        cmd = [sys.executable, "-m", "pip", "install", "raganything>=0.1.0"]
        result = subprocess.run(
            cmd,
            check=False,
            cwd=project_root,
            timeout=900,  # 15 minutes for complex package
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            # Try with --no-deps and install deps separately
            log_warning("Standard install failed, trying with --no-deps...")
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "raganything>=0.1.0",
                "--no-deps",
            ]
            result = subprocess.run(
                cmd,
                check=False,
                cwd=project_root,
                timeout=300,
                capture_output=False,
                text=True,
            )
            if result.returncode != 0:
                log_warning("raganything installation had issues")
    except Exception as e:
        log_warning(f"raganything installation error: {e}")

    # Stage 4: Install docling (alternative parser for Office/HTML documents)
    log_info("Stage 4/4: Installing docling (document parser for Office/HTML)...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "docling>=2.31.0"]
        result = subprocess.run(
            cmd,
            check=False,
            cwd=project_root,
            timeout=600,  # 10 minutes
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            log_warning("docling installation had issues (optional, can be skipped)")
    except Exception as e:
        log_warning(f"docling installation error (optional): {e}")

    # Try to install any remaining optional deps
    try:
        optional_deps = ["perplexityai>=0.1.0", "dashscope>=1.14.0"]
        for dep in optional_deps:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                check=False,
                cwd=project_root,
                timeout=120,
                capture_output=True,
                text=True,
            )
    except Exception:
        pass

    return True


def install_backend_deps(project_root: Path) -> bool:
    """Install backend dependencies"""
    log_step("Step 1/3: Installing backend dependencies")

    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        log_error(f"requirements.txt not found: {requirements_file}")
        return False

    log_info(f"Using Python: {sys.executable}")
    log_info(f"Requirements file: {requirements_file}")

    # Strategy 1: Try using uv (fastest and best resolver)
    log_info("Attempting installation with uv (recommended)...")
    if install_with_uv(requirements_file, project_root):
        log_success("Backend dependencies installed successfully with uv")
        return True

    log_warning("uv installation failed, falling back to staged pip installation...")

    # Strategy 2: Staged pip installation
    if install_with_pip_staged(requirements_file, project_root):
        log_success("Backend dependencies installed successfully with staged pip")
        return True

    # Strategy 3: Direct pip install as last resort
    log_warning("Staged installation had issues, trying direct pip install...")
    try:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_file),
        ]
        log_info("Installing backend dependencies, please wait...")

        result = subprocess.run(
            cmd,
            check=False,
            cwd=project_root,
            timeout=600,  # 10 minute timeout
            capture_output=False,  # Show output in real-time
            text=True,
        )

        if result.returncode == 0:
            log_success("Backend dependencies installed successfully")
            return True
        log_error(f"Backend dependencies installation failed (exit code: {result.returncode})")
        return False

    except subprocess.TimeoutExpired:
        log_error("Installation timeout (exceeded 10 minutes)")
        return False
    except Exception as e:
        log_error(f"Error installing backend dependencies: {e}")
        return False


def install_frontend_deps(project_root: Path) -> bool:
    """Install frontend dependencies"""
    log_step("Step 2/3: Installing frontend dependencies")

    web_dir = project_root / "web"
    package_json = web_dir / "package.json"

    if not package_json.exists():
        log_error(f"package.json not found: {package_json}")
        return False

    # Check if npm is available, if not, try to install it
    npm_path = shutil.which("npm")
    if not npm_path:
        log_warning("npm command not found")
        log_info("Attempting to install Node.js automatically...")

        installed = False
        platform = sys.platform

        # Try different installation methods based on OS
        if platform == "darwin":  # macOS
            # Try Homebrew first
            if shutil.which("brew"):
                log_info("Detected macOS with Homebrew, installing Node.js via Homebrew...")
                try:
                    result = subprocess.run(
                        ["brew", "install", "node"],
                        check=False,
                        timeout=600,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        installed = True
                        log_success("Node.js installed successfully via Homebrew")
                    else:
                        log_warning("Homebrew installation failed")
                except Exception as e:
                    log_warning(f"Homebrew installation error: {e}")

            # Try conda if Homebrew failed
            if not installed and shutil.which("conda"):
                log_info("Detected conda, installing Node.js via conda...")
                try:
                    result = subprocess.run(
                        [
                            "conda",
                            "install",
                            "-c",
                            "conda-forge",
                            "nodejs",
                            "-y",
                        ],
                        check=False,
                        timeout=600,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        installed = True
                        log_success("Node.js installed successfully via conda")
                    else:
                        log_warning("Conda installation failed")
                except Exception as e:
                    log_warning(f"Conda installation error: {e}")

        elif platform.startswith("linux"):  # Linux
            # Try apt-get (Debian/Ubuntu)
            if shutil.which("apt-get"):
                log_info("Detected Debian/Ubuntu, installing Node.js via apt...")
                try:
                    result = subprocess.run(
                        ["sudo", "apt-get", "update"],
                        check=False,
                        timeout=300,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        result = subprocess.run(
                            [
                                "sudo",
                                "apt-get",
                                "install",
                                "-y",
                                "nodejs",
                                "npm",
                            ],
                            check=False,
                            timeout=600,
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            installed = True
                            log_success("Node.js installed successfully via apt")
                        else:
                            log_warning("apt installation failed")
                    else:
                        log_warning("apt-get update failed")
                except Exception as e:
                    log_warning(f"apt installation error: {e}")

            # Try yum (RHEL/CentOS)
            if not installed and shutil.which("yum"):
                log_info("Detected RHEL/CentOS, installing Node.js via yum...")
                try:
                    result = subprocess.run(
                        ["sudo", "yum", "install", "-y", "nodejs", "npm"],
                        check=False,
                        timeout=600,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        installed = True
                        log_success("Node.js installed successfully via yum")
                    else:
                        # Try without npm (some distros bundle it with nodejs)
                        result = subprocess.run(
                            ["sudo", "yum", "install", "-y", "nodejs"],
                            check=False,
                            timeout=600,
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            installed = True
                            log_success("Node.js installed successfully via yum")
                        else:
                            log_warning("yum installation failed")
                except Exception as e:
                    log_warning(f"yum installation error: {e}")

            # Try conda if package managers failed
            if not installed and shutil.which("conda"):
                log_info("Detected conda, installing Node.js via conda...")
                try:
                    result = subprocess.run(
                        [
                            "conda",
                            "install",
                            "-c",
                            "conda-forge",
                            "nodejs",
                            "-y",
                        ],
                        check=False,
                        timeout=600,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        installed = True
                        log_success("Node.js installed successfully via conda")
                    else:
                        log_warning("Conda installation failed")
                except Exception as e:
                    log_warning(f"Conda installation error: {e}")

        elif platform == "win32":  # Windows
            # Try Chocolatey
            if shutil.which("choco"):
                log_info("Detected Windows with Chocolatey, installing Node.js via Chocolatey...")
                try:
                    result = subprocess.run(
                        ["choco", "install", "nodejs", "-y"],
                        check=False,
                        timeout=600,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        installed = True
                        log_success("Node.js installed successfully via Chocolatey")
                    else:
                        log_warning("Chocolatey installation failed")
                except Exception as e:
                    log_warning(f"Chocolatey installation error: {e}")

            # Try conda if Chocolatey failed
            if not installed and shutil.which("conda"):
                log_info("Detected conda, installing Node.js via conda...")
                try:
                    result = subprocess.run(
                        [
                            "conda",
                            "install",
                            "-c",
                            "conda-forge",
                            "nodejs",
                            "-y",
                        ],
                        check=False,
                        timeout=600,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        installed = True
                        log_success("Node.js installed successfully via conda")
                    else:
                        log_warning("Conda installation failed")
                except Exception as e:
                    log_warning(f"Conda installation error: {e}")

        # Verify installation
        if installed:
            # Update PATH and check again
            os.environ["PATH"] = (
                os.environ.get("PATH", "")
                + os.pathsep
                + "/usr/local/bin"
                + os.pathsep
                + "/opt/homebrew/bin"
            )
            npm_path = shutil.which("npm")

            if not npm_path:
                log_warning("Node.js was installed but npm is still not in PATH")
                log_info("Please restart your terminal or update PATH manually")
                log_info("Then run this script again")
                return False
        else:
            log_error("Could not automatically install Node.js")
            log_info("Please install Node.js manually using one of the following methods:")
            log_info("1. Official installer: https://nodejs.org/")
            log_info("2. Using conda: conda install -c conda-forge nodejs")
            log_info("3. Using nvm: nvm install 18 && nvm use 18")
            if platform == "darwin":
                log_info("4. Using Homebrew: brew install node")
            elif platform.startswith("linux"):
                log_info(
                    "4. Using package manager: sudo apt-get install nodejs npm (Debian/Ubuntu)"
                )
                log_info("   or: sudo yum install nodejs npm (RHEL/CentOS)")
            return False

    # Re-check npm path after potential installation
    npm_path = shutil.which("npm")
    if not npm_path:
        log_error("npm command not found after installation attempt")
        return False

    log_info(f"Using npm: {npm_path}")
    log_info(f"Frontend directory: {web_dir}")

    try:
        log_info("Installing frontend dependencies, please wait...")

        # On Windows, use shell=True to properly execute .CMD files
        # On Unix-like systems, shell=True is also safe and works correctly
        result = subprocess.run(
            "npm install",
            check=False,
            cwd=web_dir,
            shell=True,
            timeout=600,  # 10 minute timeout
            capture_output=False,  # Show output in real-time
            text=True,
        )

        if result.returncode == 0:
            log_success("Frontend dependencies installed successfully")
            return True
        log_error(f"Frontend dependencies installation failed (exit code: {result.returncode})")
        return False

    except subprocess.TimeoutExpired:
        log_error("Installation timeout (exceeded 10 minutes)")
        return False
    except Exception as e:
        log_error(f"Error installing frontend dependencies: {e}")
        return False


def verify_installation(project_root: Path) -> bool:
    """Verify installation"""
    log_step("Step 3/3: Verifying installation")

    all_ok = True

    # Check backend key packages
    log_info("Checking backend key packages...")
    backend_packages = [
        "fastapi",
        "uvicorn",
        "openai",
        "lightrag_hku",
        "raganything",
        "llama_index",
        "docling",
    ]

    for package in backend_packages:
        try:
            # Try importing package
            if package == "lightrag_hku":
                __import__("lightrag")
            elif package == "raganything":
                __import__("raganything")
            elif package == "docling":
                __import__("docling")
            else:
                __import__(package)
            log_success(f"  ✓ {package}")
        except ImportError:
            if package == "docling":
                # docling is optional
                log_warning(f"  ⚠ {package} not installed (optional, for Office/HTML parsing)")
            else:
                log_error(f"  ✗ {package} not installed")
                all_ok = False

    # Check frontend node_modules
    log_info("Checking frontend dependencies...")
    web_dir = project_root / "web"
    node_modules = web_dir / "node_modules"

    if node_modules.exists():
        # Check key packages
        key_packages = ["next", "react", "react-dom"]
        for pkg in key_packages:
            pkg_dir = node_modules / pkg
            if pkg_dir.exists():
                log_success(f"  ✓ {pkg}")
            else:
                log_error(f"  ✗ {pkg} not installed")
                all_ok = False
    else:
        log_error("  ✗ node_modules directory does not exist")
        all_ok = False

    return all_ok


def main():
    """Main function"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 DeepTutor One-Click Installation Script")
    logger.info("=" * 60)
    logger.info("This script will automatically install all frontend and backend dependencies")
    logger.info(
        "Execution flow: Backend dependencies -> Frontend dependencies -> Verify installation"
    )
    logger.info("=" * 60)

    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    log_info(f"Project root: {project_root}")

    # Check virtual environment (warning only)
    check_virtual_env()

    # Install backend dependencies
    if not install_backend_deps(project_root):
        log_error("\n❌ Backend dependencies installation failed, please check error messages")
        sys.exit(1)

    # Install frontend dependencies
    if not install_frontend_deps(project_root):
        log_error("\n❌ Frontend dependencies installation failed, please check error messages")
        sys.exit(1)

    # Verify installation
    if not verify_installation(project_root):
        log_warning(
            "\n⚠️  Some issues found during verification, but installation process completed"
        )
        log_info("If you encounter runtime errors, please check the missing packages above")
    else:
        log_success("\n✅ All dependencies installed and verified successfully!")

    # Completion message
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Installation complete!")
    logger.info("=" * 60)
    log_info("Next steps:")
    log_info("1. Configure .env file (if needed)")
    log_info("2. Start services: python scripts/start_web.py")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"\n❌ Unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
