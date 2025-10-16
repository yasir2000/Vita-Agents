"""
Version information for Vita Agents
"""

__version__ = "2.1.0"
__version_info__ = (2, 1, 0)

# Release information
__title__ = "Vita Agents"
__description__ = "Multi-Agent AI for Healthcare Interoperability"
__author__ = "Yasir Ahmed"
__author_email__ = "yasir@vita-agents.dev"
__license__ = "Apache-2.0"
__copyright__ = "Copyright 2025 Vita Agents Team"
__url__ = "https://github.com/yasir2000/vita-agents"

# Build information
__build__ = "stable"
__status__ = "Production/Stable"

# API version
__api_version__ = "v1"

# Supported versions
__python_requires__ = ">=3.9"
__fhir_versions__ = ["DSTU2", "STU3", "R4", "R5"]
__hl7_versions__ = ["2.3", "2.4", "2.5", "2.6", "2.8"]

# Feature flags
__features__ = {
    "multi_engine_fhir": True,
    "web_portal": True,
    "docker_support": True,
    "oauth2_auth": True,
    "smart_on_fhir": True,
    "performance_benchmarking": True,
    "data_migration": True,
    "cli_interface": True,
    "monitoring": True,
    "hipaa_compliance": True,
}

# Release notes URL
__release_notes__ = f"https://github.com/yasir2000/vita-agents/releases/tag/v{__version__}"

def get_version_info():
    """Get detailed version information"""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        "url": __url__,
        "build": __build__,
        "status": __status__,
        "api_version": __api_version__,
        "python_requires": __python_requires__,
        "fhir_versions": __fhir_versions__,
        "hl7_versions": __hl7_versions__,
        "features": __features__,
        "release_notes": __release_notes__,
    }

def print_version_info():
    """Print version information in a nice format"""
    info = get_version_info()
    print(f"""
🏥 {info['title']} v{info['version']}
{info['description']}

Author: {info['author']}
License: {info['license']}
Status: {info['status']}
Python: {info['python_requires']}

FHIR Versions: {', '.join(info['fhir_versions'])}
HL7 Versions: {', '.join(info['hl7_versions'])}

🌐 Homepage: {info['url']}
📄 Release Notes: {info['release_notes']}

Features:
""")
    for feature, enabled in info['features'].items():
        status = "✅" if enabled else "❌"
        feature_name = feature.replace('_', ' ').title()
        print(f"  {status} {feature_name}")

if __name__ == "__main__":
    print_version_info()