from os import path

from setuptools import find_packages, setup

curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(curdir, "requirements.txt"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="offlax",
    packages=find_packages(),
    version="0.1.0",
    license="MIT",
    description="NFNets, PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vaibhav Balloli",
    author_email="balloli.vb@gmail.com",
    url="https://github.com/vballoli/offlax",
    keywords=["offline reinforcement learning"],
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
