# my_package/setup.py
from setuptools import setup, find_packages

setup(
    name="RSP-LLM",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9.12",
    install_requires=[
        "absl-py==2.1.0",
        "decord==0.6.0",
        "grpcio==1.64.1",
        "hydra-core",
        "markdown==3.6",
        "numpy==1.26.4",
        "opencv-python",
        "openai",
        "pydantic",
        "tensorboard==2.17.0",
        "timm==0.3.2",
        "wandb",
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
        ]
    }
)
