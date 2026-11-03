"""Core geometric and mathematical primitives for FORGE."""
from __future__ import annotations
import math
from typing import NamedTuple, Tuple, List


class Vec3(NamedTuple):
    x: float
    y: float
    z: float

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec3:
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> Vec3:
        mag = self.length()
        if mag < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return self / mag

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Triangle(NamedTuple):
    a: Vec3
    b: Vec3
    c: Vec3

    @property
    def normal(self) -> Vec3:
        u = self.b - self.a
        v = self.c - self.a
        n = u.cross(v)
        return n.normalized()

    @property
    def area(self) -> float:
        u = self.b - self.a
        v = self.c - self.a
        return 0.5 * u.cross(v).length()

    def signed_volume(self) -> float:
        """Signed volume of tetrahedron formed with the origin."""
        return self.a.dot(self.b.cross(self.c)) / 6.0


def calculate_mesh_volume(triangles: List[Triangle]) -> float:
    """Calculate enclosed volume of a watertight manifold mesh via Gauss divergence theorem."""
    return sum(t.signed_volume() for t in triangles)


def calculate_mesh_surface_area(triangles: List[Triangle]) -> float:
    """Calculate total surface area of triangles."""
    return sum(t.area for t in triangles)
