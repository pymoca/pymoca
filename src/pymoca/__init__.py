# setuptools_scm will not dynamically generate the version,
# it will only do so once statically at install time.

# Getting the version dynamically is not generally recommended,
# so we only do it for editable installs. Detecting editable installs
# is a bit ugly though, see below:

import json
from importlib.metadata import Distribution

# See https://github.com/pypa/setuptools/issues/4186
# It might be possible to use a more direct way in the future with importlib-metadata >= 6.11.0
direct_url = Distribution.from_name("pymoca").read_text("direct_url.json")
pkg_is_editable = json.loads(direct_url).get("dir_info", {}).get("editable", False)

if pkg_is_editable:
    # Use dynamic way of getting version
    from setuptools_scm import get_version
    __version__ = get_version()
else:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("pymoca")
    except PackageNotFoundError:
        pass
