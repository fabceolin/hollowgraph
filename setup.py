from setuptools import setup, find_packages
import sys

print (sys.version_info)

# Conditional dependencies based on Python version
if sys.version_info >= (3, 10):
    networkx_version = "networkx==3.3"
else:
    networkx_version = "networkx==3.2.1"


setup(
    name="hollowgraph",
    version="0.0.1",
    author="Fabricio Ceolin",
    author_email="fabceolin@gmail.com",
    description="A lightweight, single-app state graph library inspired by LangGraph",
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
        "networkx==3.3",
        "pygraphviz==1.13",
    ],
    extras_require={
        "dev": ["pytest", "coverage", "hypothesis","parametrized"],
    },
)
