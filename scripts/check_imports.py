import importlib

modules = [
    'src.services.rag.factory',
    'src.services.rag.pipelines.academic',
    'src.services.rag.pipelines.lightrag',
    'src.services.rag.pipelines.llamaindex',
    'src.services.rag.pipelines.raganything',
]

for m in modules:
    try:
        mod = importlib.import_module(m)
        print(f'Imported {m} OK')
        if hasattr(mod, '__all__'):
            print('  exports:', getattr(mod, '__all__'))
    except Exception as e:
        print(f'ImportError for {m}: {e!r}')
        import traceback
        traceback.print_exc()

# Print registered pipelines
try:
    factory = importlib.import_module('src.services.rag.factory')
    print('Factory pipelines:', list(factory._PIPELINES.keys()))
except Exception as e:
    print('Factory import failed:', e)