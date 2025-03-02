"""
Setup script for Instruction Embedding Evaluation Framework.
"""

from setuptools import setup, find_packages

setup(
    name="instruction-embedding-evaluation",
    version="0.1.0",
    description="Framework for evaluating instruction embeddings in binary analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "requests>=2.27.0",
        "gensim>=4.1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)