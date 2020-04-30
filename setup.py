#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("textmap", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "textmap"
DESCRIPTION = "Tools for word and document embedding using UMAP"
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Leland McInnes"
MAINTAINER_EMAIL = "leland.mcinnes@gmail.com"
URL = "https://github.com/TutteInstitute/TextMAP"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/TutteInstitute/TextMAP"
VERSION = __version__
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "numba >= 0.49",
    "enstop >= 0.1.6",
    "umap-learn >= 0.4.2",
    "pynndescent >= 0.4.5",
    "pandas",
    "nltk",
]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov", "spacy"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
