# Backend Architecture

This document is for developers implementing a new backend or maintaining an existing one.
It explains how each backend departs from the Modelica Language Specification (MLS) and what
it does not implement. See [architecture](architecture.md) for the flattening pipeline that
feeds the backends.

## Support Across Backends

- Only the [CasADi backend](#todo--wip--stubs) handles a `ForArray`: the SymPy and XML
  backends fail with a `KeyError`

## CasADi Backend

### External Functions and Objects (MLS 12.9)

The backend compiles the external C code of a model into one shared library.

Choices the spec leaves open:

- `LibraryDirectory` is searched in `<os><bits>` (such as `linux64`) and
  `<machine>-<os>` subdirectories, then itself, but not in compiler subdirectories such
  as `win64/vs2019` (MLS 12.9.4)

### TODO / WIP / Stubs

- `min` and `max` of an empty array give the values for Real, also for Integer expressions,
  since the backend does not track the type of an expression here (MLS 10.3.4)
- An array literal with more than two dimensions raises `NotImplementedError` in an
  expression, as CasADi has no such arrays. As a variable attribute it is kept as a nested
  list, which only vector expansion (`expand_vectors`) can use
- A `ForArray` (MLS 10.3.4.1, 10.4.1) is supported with one iterator and a scalar or vector
  body, in an equation, a declaration and an attribute. The body is built once and mapped
  over the range. A constructor whose body has a range that depends on its iterator, as in
  `{sum(x[j] for j in 1:i) for i in 1:n}` or the slice in `{sum(x[i:i + 1]) for i in 1:n}`, is
  built for each value of that iterator instead.
  It raises `NotImplementedError` for several iterators, a matrix body, an implicit or
  array-valued range, a `delay()` in the body, and a subscript or range that uses the iterator
  of an enclosing for-loop, as in `for j in 1:n loop y[j] = sum(A[j, i] for i in 1:m); end for`.
  The enclosing iterator can be used as a value in the body
