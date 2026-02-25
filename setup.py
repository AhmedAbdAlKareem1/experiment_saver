from setuptools import setup, find_packages
from pathlib import Path

# Read README safely
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="experiment_saver",
    version="0.1.0",
    author="Ahmed Abd Al-Kareem",
    description="A utility to safely save Keras experiments and ROC metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AhmedAbdAlKareem1/experiment_saver",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19",
        "scikit-learn>=0.24",
    ],
    extras_require={
        "tf": ["tensorflow>=2.0"],
        "torch": ["torch>=1.0", "torchvision>=0.8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
