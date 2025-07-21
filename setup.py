"""
CodeConductor - Self-Learning Multi-Agent AI System for Intelligent Code Generation

A revolutionary AI system that combines multi-agent discussion, human approval,
and reinforcement learning for intelligent code generation.
"""

from setuptools import setup, find_packages


# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="codeconductor",
    version="2.0.0",
    author="olablom",
    author_email="olablom@example.com",
    description="Self-Learning Multi-Agent AI System for Intelligent Code Generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/olablom/CodeConductor",
    project_urls={
        "Bug Reports": "https://github.com/olablom/CodeConductor/issues",
        "Source": "https://github.com/olablom/CodeConductor",
        "Documentation": "https://github.com/olablom/CodeConductor#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
        ],
        "docker": [
            "docker>=6.0.0",
        ],
        "full": [
            "streamlit>=1.28.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "codeconductor=pipeline:main",
            "cc-pipeline=pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    keywords=[
        "ai",
        "code-generation",
        "multi-agent",
        "reinforcement-learning",
        "machine-learning",
        "software-development",
        "automation",
        "q-learning",
        "human-in-the-loop",
    ],
    license="MIT",
    zip_safe=False,
)
