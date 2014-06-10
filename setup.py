from setuptools import setup, find_packages

setup(
    name='pymola',
    install_requires=['parsimonious', 'ply', 'grako'],
    tests_require=['nose'],
    test_suite='nose.collector',
    packages=find_packages(exclude="*._test"),
)
