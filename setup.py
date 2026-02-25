from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme_path = this_dir / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="experiment-saver",  # pip name
    version="0.1.0",
    author="AhmedAbdAlKareem1",
    description="Experiment saver utilities for TensorFlow/Keras and PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AhmedAbdAlKareem1/experiment_saver",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
    extras_require={
        "tf": ["tensorflow>=2.10.0"],
        "torch": ["torch>=1.0", "torchvision"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
