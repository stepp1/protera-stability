import setuptools

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
    project_urls={
        "Bug Tracker": "https://github.com/stepp1/protera-stability/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "protera_stability"},
    packages=["."],
    python_requires=">=3.6",
)