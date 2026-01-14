"""Code generation templates and utilities."""

import os
from typing import Dict, Optional
from pathlib import Path
import re


class CodeGenerator:
    """Generate boilerplate code from templates."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
    
    def generate_route(
        self,
        name: str,
        methods: list = None,
        auth_required: bool = True
    ) -> str:
        """Generate Flask route boilerplate."""
        if methods is None:
            methods = ['GET']
        
        methods_str = ', '.join([f"'{m}'" for m in methods])
        
        auth_decorator = "@require_auth\n" if auth_required else ""
        
        template = f'''"""
{name.replace('_', ' ').title()} route.
"""

from flask import Blueprint, request, jsonify
from middleware.auth import require_auth
from utils.validators import validate_request
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('{name}', __name__)


@bp.route('/{name}', methods=[{methods_str}])
{auth_decorator}def {name}():
    """
    Handle {name.replace('_', ' ')} request.
    """
    try:
        # Validate request
        # data = validate_request(request, schema)
        
        # Process request
        result = {{
            'status': 'success',
            'data': {{}}
        }}
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error in {name}: {{e}}")
        return jsonify({{'error': str(e)}}), 500
'''
        
        return template
    
    def generate_model(
        self,
        name: str,
        fields: Dict[str, str]
    ) -> str:
        """Generate SQLAlchemy model boilerplate."""
        class_name = ''.join(word.capitalize() for word in name.split('_'))
        
        field_definitions = []
        for field_name, field_type in fields.items():
            if field_type == 'string':
                field_definitions.append(f"    {field_name} = Column(String, nullable=False)")
            elif field_type == 'int':
                field_definitions.append(f"    {field_name} = Column(Integer, nullable=False)")
            elif field_type == 'datetime':
                field_definitions.append(f"    {field_name} = Column(DateTime, nullable=False)")
            elif field_type == 'bool':
                field_definitions.append(f"    {field_name} = Column(Boolean, default=False)")
        
        fields_str = '\n'.join(field_definitions)
        
        template = f'''"""
{class_name} model.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class {class_name}(Base):
    """
    {class_name} model.
    """
    __tablename__ = '{name}'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
{fields_str}
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {{
            'id': self.id,
{self._generate_to_dict_fields(fields)}
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }}
'''
        
        return template
    
    def _generate_to_dict_fields(self, fields: Dict[str, str]) -> str:
        """Generate to_dict field mappings."""
        lines = []
        for field_name, field_type in fields.items():
            if field_type == 'datetime':
                lines.append(f"            '{field_name}': self.{field_name}.isoformat() if self.{field_name} else None,")
            else:
                lines.append(f"            '{field_name}': self.{field_name},")
        return '\n'.join(lines)
    
    def generate_test(
        self,
        name: str,
        test_type: str = 'unit'
    ) -> str:
        """Generate test boilerplate."""
        class_name = ''.join(word.capitalize() for word in name.split('_'))
        
        template = f'''"""
Tests for {name}.
"""

import pytest
from unittest.mock import Mock, patch
from {name} import {class_name}


class Test{class_name}:
    """Test {class_name} class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.instance = {class_name}()
    
    def test_initialization(self):
        """Test {class_name} initialization."""
        assert self.instance is not None
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # TODO: Implement test
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''
        
        return template
    
    def generate_cli_command(
        self,
        name: str,
        description: str
    ) -> str:
        """Generate Click CLI command boilerplate."""
        template = f'''"""
{description}
"""

import click
import logging

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """{description}"""
    pass


@cli.command()
@click.option('--option', help='Example option')
def {name}(option):
    """
    {description}
    """
    try:
        click.echo(f"Running {name}...")
        
        # TODO: Implement command logic
        
        click.echo("✓ Complete")
    
    except Exception as e:
        logger.error(f"Error in {name}: {{e}}")
        click.echo(f"Error: {{e}}", err=True)


if __name__ == '__main__':
    cli()
'''
        
        return template
    
    def generate_migration(
        self,
        name: str,
        table_name: str,
        operation: str = 'create'
    ) -> str:
        """Generate database migration boilerplate."""
        version = f"001_{name}"
        
        if operation == 'create':
            up_sql = f'''
CREATE TABLE {table_name} (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
'''
            down_sql = f"DROP TABLE {table_name};"
        
        elif operation == 'add_column':
            up_sql = f"ALTER TABLE {table_name} ADD COLUMN new_column VARCHAR(255);"
            down_sql = f"ALTER TABLE {table_name} DROP COLUMN new_column;"
        
        else:
            up_sql = "-- TODO: Add migration SQL"
            down_sql = "-- TODO: Add rollback SQL"
        
        template = f'''"""
Migration: {name}
"""

from database.migrations import Migration


migration = Migration(
    version="{version}",
    description="{name}",
    up_sql="""
{up_sql}
    """,
    down_sql="""
{down_sql}
    """
)
'''
        
        return template
    
    def generate_dockerfile(
        self,
        base_image: str = "python:3.10-slim",
        port: int = 8080
    ) -> str:
        """Generate Dockerfile boilerplate."""
        template = f'''# Multi-stage build for CHIMERA Platform

FROM {base_image} AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM {base_image}

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT={port}

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:{port}/health')"

# Run application
CMD ["python", "app.py"]
'''
        
        return template
    
    def save_file(self, content: str, filepath: str):
        """Save generated code to file."""
        full_path = self.base_path / filepath
        
        # Create directory if it doesn't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(full_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Generated: {filepath}")


class TemplateEngine:
    """Template engine for code generation."""
    
    @staticmethod
    def render(template: str, context: Dict) -> str:
        """Render template with context variables."""
        result = template
        
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        
        return result
    
    @staticmethod
    def load_template(template_path: str) -> str:
        """Load template from file."""
        with open(template_path, 'r') as f:
            return f.read()


# CLI interface
if __name__ == '__main__':
    import click
    
    @click.group()
    def cli():
        """Code generation CLI."""
        pass
    
    @cli.command()
    @click.argument('name')
    @click.option('--methods', default='GET', help='HTTP methods (comma-separated)')
    @click.option('--auth/--no-auth', default=True, help='Require authentication')
    def route(name, methods, auth):
        """Generate a new route."""
        generator = CodeGenerator()
        methods_list = [m.strip() for m in methods.split(',')]
        
        code = generator.generate_route(name, methods_list, auth)
        filepath = f"routes/{name}.py"
        
        generator.save_file(code, filepath)
    
    @cli.command()
    @click.argument('name')
    @click.option('--fields', help='Fields (name:type,name:type)')
    def model(name, fields):
        """Generate a new model."""
        generator = CodeGenerator()
        
        fields_dict = {}
        if fields:
            for field in fields.split(','):
                field_name, field_type = field.split(':')
                fields_dict[field_name.strip()] = field_type.strip()
        
        code = generator.generate_model(name, fields_dict)
        filepath = f"models/{name}.py"
        
        generator.save_file(code, filepath)
    
    @cli.command()
    @click.argument('name')
    def test(name):
        """Generate a new test."""
        generator = CodeGenerator()
        
        code = generator.generate_test(name)
        filepath = f"tests/test_{name}.py"
        
        generator.save_file(code, filepath)
    
    cli()