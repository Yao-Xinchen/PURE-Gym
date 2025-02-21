from setuptools import find_packages
from distutils.core import setup

setup(
    name="pure_gym",
    version="1.0.0",
    author="Xinchen Yao",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="yao29@illinois.ed",
    description="Isaac Gym environments for Bibot",
    install_requires=[
        "torch",
        "isaacgym",
        "matplotlib",
        "tensorboard",
        "setuptools==59.5.0",
        "numpy>=1.16.4",
        "numpy<1.20.0",
        "GitPython",
        "onnx",
    ],
)
