# <img alt="Pymoca" src="branding/icons/pymocalogo.svg" height="60">

A python/modelica based simulation environment.

[![Travis](https://img.shields.io/travis/pymoca/pymoca/master.svg?label=Travis%20CI)](https://travis-ci.org/pymoca/pymoca)
[![AppVeyor](https://img.shields.io/appveyor/ci/pymoca/pymoca/master.svg?label=AppVeyor)](https://ci.appveyor.com/project/pymoca/pymoca)
[![Coverage Status](https://img.shields.io/coveralls/pymoca/pymoca/master.svg)](https://coveralls.io/r/pymoca/pymoca)


## Install

1. Install anaconda 3.

2. Setup environment.

```bash
./create_conda_env.sh enduser
. activate pymoca
jupyter notebook
```

## Examples
* [Sympy Example](test/notebooks/Spring.ipynb)
* [Casadi Example](test/notebooks/Casadi.ipynb)

## Roadmap

### Completed Tasks

* hello world compilation
* Project setup.
* Unit testing for parsers.
* Parsing basic hello world example.
* Travis continuous integration testing setup.
* Coveralls coverage testing setup.

### TODO

* add more complicated test cases
* resolve grammar issues
* support more of modelica language
* add modelica magic support for ipython?

<!--- vim:ts=4:sw=4:expandtab:
!-->
