from setuptools import setup, find_packages

setup(
    name="experiment_saver",
    version="0.1.0",
    author="Ahmed Abd Al-Kareem",
    description="A utility for saving TensorFlow/Keras experiments and binary ROC metrics.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AhmedAbdAlKareem1/experiment_saver",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "tensorflow>=2.10.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)