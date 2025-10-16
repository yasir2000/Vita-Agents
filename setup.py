#!/usr/bin/env python3
"""
Setup script for Vita Agents - Multi-Agent AI for Healthcare Interoperability
"""

import os
import re
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from __init__.py
def get_version():
    """Extract version from vita_agents/__init__.py"""
    init_file = this_directory / "vita_agents" / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding='utf-8')
        match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content)
        if match:
            return match.group(1)
    return "2.1.0"  # Fallback version

# Read requirements
def get_requirements(filename):
    """Read requirements from requirements file"""
    requirements_file = this_directory / filename
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

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


# Core requirements
install_requires = read_requirements()

# Optional requirements for different features
extras_require = {
    'fhir-engines': [
        'aiohttp>=3.8.0',
        'structlog>=23.0.0',
        'rich>=13.0.0',
        'click>=8.0.0',
        'pydantic>=2.0.0',
    ],
    'web-portal': [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'jinja2>=3.1.0',
        'python-multipart>=0.0.6',
        'websockets>=12.0',
    ],
    'docker': [
        'docker>=6.1.0',
        'docker-compose>=1.29.0',
    ],
    'monitoring': [
        'prometheus-client>=0.19.0',
        'grafana-api>=1.0.3',
        'psutil>=5.9.0',
    ],
    'security': [
        'cryptography>=41.0.0',
        'pyjwt>=2.8.0',
        'bcrypt>=4.1.0',
        'python-jose[cryptography]>=3.3.0',
    ],
    'testing': [
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.1.0',
        'pytest-xdist>=3.3.0',
        'pytest-timeout>=2.2.0',
        'pytest-benchmark>=4.0.0',
        'locust>=2.17.0',
        'factory-boy>=3.3.0',
        'faker>=20.0.0',
    ],
    'dev': read_dev_requirements() + [
        'bandit>=1.7.5',
        'safety>=2.3.0',
        'pip-audit>=2.6.0',
        'pre-commit>=3.5.0',
    ],
    'docs': [
        'mkdocs>=1.5.0',
        'mkdocs-material>=9.4.0',
        'mkdocs-mermaid2-plugin>=1.1.0',
        'sphinx>=7.2.0',
        'sphinx-rtd-theme>=1.3.0',
        'sphinx-autodoc-typehints>=1.25.0',
    ],
}

# Add 'all' option that includes everything
extras_require['all'] = list(set().union(*extras_require.values()))
extras_require['all'] = list(set().union(*extras_require.values()))

setup(
    name="vita-agents",
    version=get_version(),
    author="Yasir Ahmed",
    author_email="yasir@vita-agents.dev",
    description="Multi-Agent AI for Healthcare Interoperability - FHIR, HL7, and EHR Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yasir2000/vita-agents",
    project_urls={
        "Homepage": "https://vita-agents.org",
        "Documentation": "https://vita-agents.readthedocs.io",
        "Repository": "https://github.com/yasir2000/vita-agents",
        "Bug Tracker": "https://github.com/yasir2000/vita-agents/issues",
        "Changelog": "https://github.com/yasir2000/vita-agents/blob/main/CHANGELOG.md",
        "Discord": "https://discord.gg/vita-agents",
        "LinkedIn": "https://linkedin.com/company/vita-agents",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    package_data={
        "vita_agents": [
            "config/*.yml",
            "config/*.yaml",
            "config/*.json",
            "templates/*.html",
            "templates/*.jinja2",
            "static/css/*.css",
            "static/js/*.js",
            "static/images/*",
            "schemas/*.json",
            "schemas/*.xsd",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Web Environment",
        "Environment :: Console",
        "Framework :: FastAPI",
        "Framework :: AsyncIO",
        "Natural Language :: English",
        "Typing :: Typed",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "vita-agents=vita_agents.cli.main:main",
            "fhir-engines=vita_agents.cli.fhir_engines_cli:cli",
            "vita-portal=vita_agents.web.portal:main",
            "vita-orchestrator=vita_agents.orchestrator.main:main",
        ],
    },
    keywords=[
        "healthcare", "FHIR", "HL7", "EHR", "interoperability", 
        "medical", "agents", "AI", "machine learning", "clinical",
        "HIPAA", "compliance", "medical records", "health informatics",
        "telemedicine", "telehealth", "clinical decision support",
        "population health", "public health", "medical imaging",
        "DICOM", "SNOMED", "ICD-10", "LOINC", "CPT"
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