from numpy import allclose
from orix.vector import Vector3d
from orix.data import sdss_austenite
from filelock import FileLock

import pytest


def vec_eq(v1: Vector3d, v2: Vector3d) -> bool:
    return allclose(v1.data, v2.data)


@pytest.fixture
def x() -> Vector3d:
    return Vector3d.xvector()


@pytest.fixture
def y() -> Vector3d:
    return Vector3d.yvector()


@pytest.fixture
def z() -> Vector3d:
    return Vector3d.zvector()


# Adapt https://github.com/pytest-dev/pytest/issues/1402#issuecomment-186299177
# to avoid pytest issues with attempting to download the dataset multiple times


def pytest_configure(config):
    print("Start")
    if is_master(config):
        sdss_austenite(allow_download=True)


def is_master(config):
    return hasattr(config, "workerinput")


# Kinda hacky, but this ensures only a single download for the whole (parallel) session.
# Taken from the xdist documentation:
# https://pytest-xdist.readthedocs.io/en/latest/how-to.html#making-session-scoped-fixtures-execute-only-once
@pytest.fixture(scope="session")
def default_xmap(tmp_path_factory, worker_id):
    if worker_id == "master":
        return sdss_austenite(allow_download=True)

    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "default_xmap.tmp"

    with FileLock(str(fn) + ".lock"):
        if fn.is_file():
            data = sdss_austenite(allow_download=False)
        else:
            data = sdss_austenite(allow_download=True)
            fn.write_text("Download complete")
    return data
