# pymola

A python/modelica based simulation environment.

[![Build Status](https://travis-ci.org/pymola/pymola.svg)](https://travis-ci.org/pymola/pymola)
[![Build status](https://ci.appveyor.com/api/projects/status/vqoax3jkqsv578u7?svg=true)](https://ci.appveyor.com/project/pymola/pymola)
[![Coverage Status](https://img.shields.io/coveralls/pymola/pymola.svg)](https://coveralls.io/r/pymola/pymola)

## Install

1. Install anaconda 3.

2. Setup environment.

```bash
./create_conda_env.sh enduser
. activate pymola
jupyter notebook
```

## Examples
[Sympy Example](test/Spring.ipynb)
[Casadi Example](test/Casadi.ipynb)

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
