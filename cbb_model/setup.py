#!/usr/bin/env python3
"""
Setup script for CBB Betting ML System.

This script allows the package to be installed via pip.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CBB Betting ML System - A comprehensive machine learning system for College Basketball betting analysis."

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="cbb-betting-ml",
    version="1.0.0",
    description="A comprehensive machine learning system for College Basketball betting analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="CBB Betting ML Team",
    author_email="team@example.com",
    url="https://github.com/example/cbb-betting-ml",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="basketball, betting, machine learning, sports analytics, ncaa",
    project_urls={
        "Bug Reports": "https://github.com/example/cbb-betting-ml/issues",
        "Source": "https://github.com/example/cbb-betting-ml",
        "Documentation": "https://github.com/example/cbb-betting-ml#readme",
    },
    entry_points={
        "console_scripts": [
            "cbb-scrape-games=cbb_model.src.scrape_games:main",
            "cbb-scrape-odds=cbb_model.src.scrape_odds:main",
            "cbb-etl=cbb_model.src.etl:main",
        ],
    },
    zip_safe=False,
)