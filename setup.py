from setuptools import find_packages
from setuptools import setup


setup(
    name='torch-transformations',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[],  # see requirements.txt
    author='Dorian Henning & Simon Schaefer',
    author_email='simon.k.schaefer@gmail.com',
    license='MIT',
    url='https://github.com/dorianhenning/pytorch_transformation',
    description='PyTorch transformations in a fully differentiable way.',
)
