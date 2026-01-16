"""
Utility to access standard library logging module.

This module exists because we have a package named 'src/logging' which shadows
Python's built-in 'logging' module. We need to access the stdlib logging from
within our logging package.

We use __import__  with fromlist to force absolute import of the stdlib logging.
"""

# Force absolute import of stdlib logging module (not relative)
stdlib_logging = __import__('logging', fromlist=[''], level=0)
