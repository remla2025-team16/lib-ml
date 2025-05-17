from setuptools import setup, find_packages

setup(
    name="lib-ml",
    version="0.1.0",
    author="Anyan Huang",
    author_email="anyanhuang@tudelft.nl",
    description="A library for machine learning preprocessing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/remla2025-team16/lib-ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "scikit-learn",
    ],
)