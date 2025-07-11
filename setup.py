from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="soccer-agent",
    version="1.0.0",
    author="Soccer Agent Team",
    author_email="contact@socceragent.com",
    description="An intelligent agent for comprehensive soccer player analysis using StatsBomb data, ML, and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/soccer-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
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
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "soccer-agent=soccer_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "soccer_agent": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="soccer football analysis statsbomb machine-learning llm openai langchain",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/soccer-agent/issues",
        "Source": "https://github.com/yourusername/soccer-agent",
        "Documentation": "https://github.com/yourusername/soccer-agent#readme",
    },
) 