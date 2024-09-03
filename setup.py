from setuptools import setup, find_packages

setup(
    name="sololgraph",
    version="0.0.1",
    author="Fabricio Ceolin",
    author_email="fabceolin@gmail.com",
    description="A lightweight, single-thread state graph library inspired by LangGraph",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fabceolin/sololgraph",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "networkx",
        "graphviz",
    ],
    extras_require={
        "dev": ["pytest", "coverage"],
    },
)
