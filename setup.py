import sys

import setuptools

sys.path.insert(0, "src")
import protera_stability

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="protera_stability",
    version="0.0.1",
    author="Victor Faraggi",
    author_email="victor.faraggi@ug.uchile.cl",
    description="Tools used for the Protein Stability Prediction project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stepp1/protera-stability",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "fsspec",
        "h5py",
        "pandas",
        "numpy",
        "tqdm",
        "torch >= 1.6.0",
        "torchmetrics",
        "pytorch_lightning",
        "scikit-learn",
        "joblib",
    ],
)
