from orix.vector import Vector3d
from numpy import allclose

def vec_eq(v1: Vector3d, v2: Vector3d) -> bool:
    return allclose(v1.data, v2.data)

x = Vector3d.xvector()
y = Vector3d.yvector()
z = Vector3d.zvector()

from orix.data import sdss_austenite
def get_xmap(): 
    return sdss_austenite(allow_download=True)
