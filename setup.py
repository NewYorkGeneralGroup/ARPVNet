from setuptools import setup, find_packages

setup(
    name="arpvnet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.2",
    ],
)
