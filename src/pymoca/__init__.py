from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pymoca")
except PackageNotFoundError:
    pass
