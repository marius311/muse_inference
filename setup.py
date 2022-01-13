
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description  =  ""#(here / "README.md").read_text(encoding = "utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name = "muse_inference", 
    version = "1.0.0",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/marius311/muse_inference",
    author = "Marius Millea", 
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",

        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",

        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by "pip install". See instead "python_requires" below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords = "sample, setuptools, development",  # Optional
    packages = ["muse_inference"],
    python_requires = ">=3.6, <4",
    install_requires = ["numpy","scipy","tqdm"], 
    extras_require = {
        "jax": ["jax"],
        "pymc": ["pymc>=4.0.0b1"],
    },
)
