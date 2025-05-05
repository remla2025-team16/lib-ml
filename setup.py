from setuptools import setup, find_packages

setup(
    name="libml",
    version="0.1.0",  # Will be overridden by GitHub tag
    description="Reusable preprocessing library for sentiment analysis",
    author="Gyum Cho",
    packages=find_packages(),
    install_requires=[
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
)
