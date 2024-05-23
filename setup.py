from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pIC50_prediction',
    version='0.1.0',
    packages=find_packages(),
    description='A brief description of the package',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/comp-med-research/pIC50_prediction',
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    install_requires=required,
    dependency_links=['https://download.pytorch.org/whl/cu113'],
)