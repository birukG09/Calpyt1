"""
Setup script for Calpyt1 - Comprehensive Python Mathematical Computing Framework
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="calpyt1",
    version="0.1.0",
    author="Calpyt1 Development Team",
    author_email="dev@calpyt1.org",
    description="A comprehensive Python mathematical computing framework for calculus applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/calpyt1/calpyt1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sympy>=1.11",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "sphinx>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "notebooks": [
            "jupyterlab>=3.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "calpyt1=calpyt1.cli.main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/calpyt1/calpyt1/issues",
        "Source": "https://github.com/calpyt1/calpyt1",
        "Documentation": "https://calpyt1.readthedocs.io/",
    },
)
