"""Compute the version number and store it in the `__version__` variable.

Based on <https://github.com/maresb/hatch-vcs-footgun-example>.
"""


def _get_hatch_version():
    """Compute the most up-to-date version number in a development environment.

    Returns `None` if Hatchling is not installed, e.g. in a production environment.

    For more details, see <https://github.com/maresb/hatch-vcs-footgun-example/>.
    """
    import os

    try:
        import hatch_vcs  # noqa: F401
        from hatchling.metadata.core import ProjectMetadata
        from hatchling.plugin.manager import PluginManager
        from hatchling.utils.fs import locate_file
    except ImportError:
        # Hatchling and/or hatch-vcs are not installed, so probably we are not
        # in a development environment.
        return None

    pyproject_toml = locate_file(__file__, "pyproject.toml")
    if pyproject_toml is None:
        raise RuntimeError("pyproject.toml not found although hatchling is installed")
    root = os.path.dirname(pyproject_toml)
    metadata = ProjectMetadata(root=root, plugin_manager=PluginManager())
    # Version can be either statically set in pyproject.toml or computed dynamically:
    return metadata.core.version or metadata.hatch.version.cached


def _get_importlib_metadata_version():
    """Compute the version number using importlib.metadata.

    This is the official Pythonic way to get the version number of an installed
    package. However, it is only updated when a package is installed. Thus, if a
    package is installed in editable mode, and a different version is checked out,
    then the version number will not be updated.
    """
    from importlib.metadata import version

    __version__ = version(__package__)
    return __version__


__version__ = _get_hatch_version() or _get_importlib_metadata_version()
