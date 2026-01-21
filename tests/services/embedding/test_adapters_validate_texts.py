# -*- coding: utf-8 -*-
import importlib.util
import pathlib
import pytest

# Load the local base module directly to avoid importing the package which may
# have heavyweight side-effects during test collection.
BASE_DIR = pathlib.Path(__file__).resolve().parents[3] / "src" / "services" / "embedding" / "adapters"
BASE = BASE_DIR

# Create synthetic package entries so relative imports inside adapter modules
# (e.g., from .base import ...) resolve correctly when we load by file path.
import sys
import types

if "src" not in sys.modules:
    sys.modules["src"] = types.ModuleType("src")
if "src.services" not in sys.modules:
    sys.modules["src.services"] = types.ModuleType("src.services")
if "src.services.embedding" not in sys.modules:
    emb_pkg = types.ModuleType("src.services.embedding")
    emb_pkg.__path__ = [str(BASE_DIR.parent)]
    sys.modules["src.services.embedding"] = emb_pkg
if "src.services.embedding.adapters" not in sys.modules:
    adapters_pkg = types.ModuleType("src.services.embedding.adapters")
    adapters_pkg.__path__ = [str(BASE_DIR)]
    sys.modules["src.services.embedding.adapters"] = adapters_pkg

# Load base module into the package namespace
spec_base = importlib.util.spec_from_file_location(
    "src.services.embedding.adapters.base", str(BASE_DIR / "base.py")
)
base_mod = importlib.util.module_from_spec(spec_base)
spec_base.loader.exec_module(base_mod)  # type: ignore
sys.modules["src.services.embedding.adapters.base"] = base_mod
EmbeddingRequest = getattr(base_mod, "EmbeddingRequest")

async def _load_and_invoke_embed(module_path: pathlib.Path, config: dict):
    # Load module under the package name so relative imports work
    full_name = f"src.services.embedding.adapters.{module_path.stem}"
    spec = importlib.util.spec_from_file_location(full_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    # Load module in isolation
    spec.loader.exec_module(module)  # type: ignore

    # Get adapter class (assumes one adapter class per module following naming convention)
    # e.g., ollama -> OllamaEmbeddingAdapter
    class_name = ''.join(part.capitalize() for part in module_path.stem.split('_')) + 'EmbeddingAdapter'
    adapter_cls = getattr(module, class_name)

    adapter = adapter_cls(config)

    request = EmbeddingRequest(texts=[], model=config.get("model", "test"))

    with pytest.raises(ValueError):
        await adapter.embed(request)


@pytest.mark.asyncio
async def test_ollama_validates_texts():
    await _load_and_invoke_embed(BASE / "ollama.py", {"base_url": "http://localhost"})


@pytest.mark.asyncio
async def test_openai_compatible_validates_texts():
    await _load_and_invoke_embed(BASE / "openai_compatible.py", {"base_url": "http://localhost"})


@pytest.mark.asyncio
async def test_cohere_validates_texts():
    await _load_and_invoke_embed(BASE / "cohere.py", {"base_url": "http://localhost", "api_key": "x"})


@pytest.mark.asyncio
async def test_jina_validates_texts():
    await _load_and_invoke_embed(BASE / "jina.py", {"base_url": "http://localhost", "api_key": "x"})


@pytest.mark.asyncio
async def test_azure_validates_texts(monkeypatch):
    # Azure adapter imports openai at module import time and instantiates a client in __init__.
    # Patch the SDK class before instantiating to avoid network or heavy imports.
    module_path = BASE / "azure.py"
    full_name = f"src.services.embedding.adapters.{module_path.stem}"
    spec = importlib.util.spec_from_file_location(full_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    # Create a fake SDK class to satisfy AzureEmbeddingAdapter.__init__
    class FakeClient:
        def __init__(self, **kwargs):
            pass

    # Execute module under package name so relative imports resolve
    sys.modules[full_name] = module
    spec.loader.exec_module(module)  # type: ignore
    setattr(module, "openai", type("_", (), {"AsyncAzureOpenAI": FakeClient}))

    adapter_cls = getattr(module, "AzureEmbeddingAdapter")

    adapter = adapter_cls({"api_key": "x", "base_url": "http://localhost"})

    request = EmbeddingRequest(texts=[], model="test")

    with pytest.raises(ValueError):
        await adapter.embed(request)
