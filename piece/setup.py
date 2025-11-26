from setuptools import setup, find_packages

setup(
    name="piece",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
    ],
    python_requires=">=3.8",
)