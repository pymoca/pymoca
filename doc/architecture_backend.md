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

The backend compiles the external C code of a model into one shared library and calls it
numerically through CasADi callbacks. External objects are ids in the CasADi graph. An
external object bound by a function other than its constructor, or not bound at all, raises
an error (MLS 12.9.7).

Choices the spec leaves open:

- A model's external objects are constructed once, at generation time, from the values
  of their constructor arguments then, and live as long as the `Model` (MLS 12.9.7 only
  requires construction before first use). So these arguments must evaluate at generation
  time, and a parameter changed on the `Model` afterwards does not reach the object
- A `zeroDerivative` annotation is only used for a call whose inputs it names are
  numeric constants there. A parameter counts as varying, since sensitivities may be
  taken with respect to it (MLS 12.7.1)
- `builtin` functions are compiled callbacks like `C` functions (MLS 12.9 leaves them to
  the tool). A model calling, for example, `Modelica.Math.sin` needs a C compiler, gets
  finite-difference sensitivities, and cannot be code generated or cached
- `LibraryDirectory` is searched in `<os><bits>` (such as `linux64`) and
  `<machine>-<os>` subdirectories, then itself, but not in compiler subdirectories such
  as `win64/vs2019` (MLS 12.9.4)
- Function-local objects are destroyed in construction order, also when one was built
  from another (MLS 12.9.7 does not fix the order)

Departures from the spec:

- `impure` external functions, and the deprecated ones declared neither `pure` nor
  `impure`, are called like pure functions in equations, so CasADi may merge or repeat
  their calls. A warning is given instead of the diagnostics of MLS 12.3
- An equation `obj = Constructor(...)` in an equation section binds an external object,
  which MLS 12.9.7 only allows in its declaration
- A Boolean array output is not normalized to 0 and 1, and an enumeration output is not
  checked to be in range (MLS 12.9.1.1)

Not implemented:

- Named arguments are dropped by the parser, so a call with named arguments gets the
  defaults of its trailing inputs, or fails when they have none, such as the MLS 12.9.7
  example `MyTable(fileName = "testTables.txt", tableName = "table1")` (MLS 12.4.1)
- Default inputs are only filled in for calls evaluated at generation time. A call in an
  equation must give all inputs of an external function
- A String argument of an external function must evaluate at generation time, from
  literals and String parameters. A String input of a calling Modelica function raises
  `NotImplementedError`, and a Modelica function cannot be called in an equation with a
  String argument
- Language `"FORTRAN 77"` raises `NotImplementedError` (MLS 12.9)
- Arrays with more than two dimensions as arguments of an external function called in an
  equation

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
