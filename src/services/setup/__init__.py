"""
Setup Service
=============

System setup and initialization for DeepTutor.

Usage:
    from src.services.setup import init_user_directories, get_backend_port
    
    # Initialize user directories
    init_user_directories()
    
    # Get server ports
    backend_port = get_backend_port()
    frontend_port = get_frontend_port()
"""

from .init import (
    init_user_directories,
    get_backend_port,
    get_frontend_port,
    get_ports,
    print_port_config_tutorial,
)

__all__ = [
    "init_user_directories",
    "get_backend_port",
    "get_frontend_port",
    "get_ports",
    "print_port_config_tutorial",
]

