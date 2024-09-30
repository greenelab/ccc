import setuptools

# Commands to publish new package:
#
# rm -rf dist/
# python setup.py sdist
# twine upload dist/*

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ccc-coef",
    version="0.2.2",  # remember to change libs/ccc/__init__.py file also
    author="Milton Pividori",
    author_email="miltondp@gmail.com",
    description="The Clustermatch Correlation Coefficient (CCC) is a highly-efficient, next-generation not-only-linear correlation coefficient that can work on numerical and categorical data types.",
    license="BSD-2-Clause Plus Patent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/greenelab/ccc",
    package_dir={"": "libs"},
    packages=[
        "ccc/coef",
        "ccc/numpy",
        "ccc/pytorch",
        "ccc/scipy",
        "ccc/sklearn",
        "ccc/utils",
    ],
    python_requires=">=3.9",
    install_requires=[
        # numpy.typing is only available in numpy>=1.21.0
        "numpy>=1.21.0",
        "scipy",
        "numba",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
    ],
)
