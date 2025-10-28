from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="raw-fusion-diffusion",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-Time Multi-Frame RAW Fusion with Diffusion-Guided Detail Recovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/raw-fusion-diffusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "raw-fusion=main:main",
        ],
    },
)
