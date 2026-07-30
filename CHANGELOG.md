# Changelog

This is a summary of user-facing changes in each pymoca release version.
For full release notes including change details, see the
[Pymoca release list on GitHub](https://github.com/pymoca/pymoca/releases/).

<!-- When tagging a release, add a section with the heading `## <version>`
and a summary of the user-facing changes. See the existing entries.

The changelog job in `.github/workflows/ci.yml` extracts the section matching
the pushed tag, and the deploy job uses it as the release summary on GitHub,
above the auto-generated pull request list.

A tag with no section here fails before anything is published. A `.devN` or
`rcN` tag checks the section for the release it leads up to, so `0.12.0.dev1`
or `0.12.0rc1` needs a `## 0.12.0` heading.
-->

## 0.12.0

Pymoca now contains a complete rewrite of flattening based on the Modelica Language Specification v3.5, tested against 645 example models in Modelica Standard Library 4.0.x. 577 of them (about 90%) flatten without error. Every remaining failure is one of two unimplemented features, `ExternalObject` and stream connectors. Getting those models through the backends is separate work still in progress.

Pymoca also now resolves top-level names over MODELICAPATH, following the ordered first-wins precedence of MLS section 13.3.

A `pymoca` script is installed into your PATH with the PyPI install. Run `pymoca --help` to see how to use it.

This release makes some minor breaking API changes for pymoca 0.11 users, including a `tree.flatten_class()` that is a different function than the 0.11 helper of the same name. See [doc/index.md](https://github.com/pymoca/pymoca/blob/a12e8813eed8caa66e12d59e26c348bc6035dbea/doc/index.md) for a list of differences with pymoca 0.11 and links to documentation on installation, usage, examples, and the new flattening architecture which includes a list of yet unimplemented Modelica features. Python 3.10+ is now required. The ModelicaXML backend is unmaintained and may be removed in a future release.

## Earlier Versions

See [GitHub Releases](https://github.com/pymoca/pymoca/releases).
