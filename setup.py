from setuptools import setup, find_packages

setup(
    name='pyedpyper',
    version=0.01,

    url='https://github.com/lukas-hager/pyedpyper',
    author='Lukas Hager',
    author_email='lghhager@uw.edu',

    install_requires = [
        'numpy',
        'pandas',
        'matplotlib'
    ],

    packages=find_packages(),
)