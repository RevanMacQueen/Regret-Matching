import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rm",
    version="0.0.1",
    author="Revan MacQueen",
    author_email="revan@ualberta.ca",
    description="Implementation of Regret-Matching Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RevanMacQueen/SelfPlay",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "rm"},
    packages=setuptools.find_packages(where="rm"),
    python_requires=">=3.8",
)