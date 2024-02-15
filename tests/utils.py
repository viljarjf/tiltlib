from numpy import allclose
from orix.vector import Vector3d
from orix.data import sdss_austenite

from pytest import fixture

def vec_eq(v1: Vector3d, v2: Vector3d) -> bool:
    return allclose(v1.data, v2.data)

@fixture
def x() -> Vector3d:
    return Vector3d.xvector()
@fixture
def y() -> Vector3d:
    return Vector3d.yvector()
@fixture
def z() -> Vector3d:
    return Vector3d.zvector()

# Adapt https://github.com/pytest-dev/pytest/issues/1402#issuecomment-186299177
# to avoid pytest issues with attempting to download the dataset multiple times

def pytest_configure(config):        
    if is_master(config):
        sdss_austenite(allow_download=True)

def is_master(config):
    return hasattr(config, "workerinput")

@fixture
def default_xmap():
    return sdss_austenite(allow_download=False)
