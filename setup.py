import os
from setuptools import setup, find_packages

def parse_requirements(file):
    return [line for line in open(os.path.join(os.path.dirname(__file__), file))]

setup(
    name='slr',
    python_requires='>=3.6',
    packages=find_packages(include=['slr', 'slr.*']),
    install_requires=parse_requirements('requirements.txt')
)
