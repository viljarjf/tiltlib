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


# Adapt https://stackoverflow.com/questions/17801300/how-to-run-a-method-before-all-tests-in-all-classes
# to avoid pytest issues with attempting to download the dataset multiple times


def pytest_sessionstart(session):
    sdss_austenite(allow_download=True)


# Kinda hacky, but this ensures only a single download for the whole (parallel) session.
# Taken from the xdist documentation:
# https://pytest-xdist.readthedocs.io/en/latest/how-to.html#making-session-scoped-fixtures-execute-only-once
@pytest.fixture(scope="session")
def default_xmap():
    return sdss_austenite(allow_download=False)
