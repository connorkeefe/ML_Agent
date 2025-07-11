from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-processor-mcp",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MCP-based ML Processing System with Distributed Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-processor-mcp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "full": ["xgboost", "lightgbm", "catboost"],
    },
    entry_points={
        "console_scripts": [
            "ml-processor=orchestrator.main:main",
            "start-servers=scripts.start_servers:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mcp_servers": ["*/resources/*.json"],
    },
)