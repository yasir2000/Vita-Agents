"""
Setup script for Vita Agents.
Supports: python setup.py install
"""

from setuptools import setup, find_packages
import os


def read_requirements():
    """Read requirements from pyproject.toml dependencies."""
    # Basic requirements for healthcare interoperability
    return [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.13.0",
        "asyncpg>=0.29.0",
        "redis>=5.0.0",
        "celery>=5.3.0",
        "crewai>=0.28.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "openai>=1.3.0",
        "anthropic>=0.8.0",
        "fhirclient>=4.1.0",
        "hl7apy>=1.3.4",
        "pydicom>=2.4.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "python-multipart>=0.0.6",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "typer>=0.9.0",
        "rich>=13.7.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "structlog>=23.2.0",
        "prometheus-client>=0.19.0",
        "cryptography>=41.0.0",
        "httpx>=0.25.0",
    ]


def read_dev_requirements():
    """Read development requirements."""
    return [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "black>=23.11.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.7.0",
        "pre-commit>=3.6.0",
        "httpx>=0.25.0",
    ]


def read_long_description():
    """Read long description from README."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Multi-Agent AI Framework for Healthcare Interoperability"


setup(
    name="vita-agents",
    version="1.0.0",
    author="Yasir",
    author_email="yasir@vita-agents.org",
    description="Multi-Agent AI Framework for Healthcare Interoperability",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yasir2000/vita-agents",
    project_urls={
        "Bug Tracker": "https://github.com/yasir2000/vita-agents/issues",
        "Documentation": "https://vita-agents.org/docs",
        "Source Code": "https://github.com/yasir2000/vita-agents",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": read_dev_requirements(),
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
            "mkdocs-mermaid2-plugin>=1.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "vita-agents=vita_agents.cli.main:main",
            "vita-orchestrator=vita_agents.core.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vita_agents": [
            "config/*.yml",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    keywords=[
        "healthcare", "fhir", "hl7", "ehr", "ai", "agents", 
        "interoperability", "hipaa", "medical", "clinical"
    ],
    license="Apache 2.0",
    zip_safe=False,
)