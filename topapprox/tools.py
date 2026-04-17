"""Miscellaneous helpers used in notebooks and tests."""

from __future__ import annotations

import random


class Tools:
    """Namespace for package-level helper utilities."""

    @staticmethod
    def gwf_generator(
        n_faces: int,
        *,
        face_size_range: tuple[int, int] = (3, 12),
        n_holes: int = 1,
        unique_faces: bool = True,
    ) -> tuple[list[list[int]], list[list[int]], int]:
        """Generate a random planar graph-with-faces instance."""

        min_size, max_size = face_size_range
        n_vertices = random.randint(min_size, max_size)
        boundary = list(range(n_vertices))
        faces = [boundary.copy()]
        holes: list[list[int]] = []
        min_new_vertices = 1 if unique_faces else 0

        for _ in range(n_faces - 1):
            boundary_length = len(boundary)
            start = random.randint(0, boundary_length - 1)
            increment = random.randint(2, min(boundary_length - 1, max_size - min_new_vertices))
            end = start + increment
            new_vertex_count = random.randint(
                max(min_new_vertices, min_size - increment),
                max_size - increment,
            )
            new_segment = list(range(n_vertices, n_vertices + new_vertex_count))

            face = Tools._wrap_slice(boundary, start, end) + new_segment
            faces.append(face)
            n_vertices += new_vertex_count

            boundary = Tools._wrap_slice(boundary, end - 1, start + 1) + list(reversed(new_segment))

        faces.append(boundary.copy())

        for _ in range(min(n_holes, len(faces))):
            holes.append(faces.pop(random.randrange(len(faces))))

        return faces, holes, n_vertices

    @staticmethod
    def _wrap_slice(values: list[int], start: int, end: int) -> list[int]:
        size = len(values)
        start %= size
        end %= size
        if start < end:
            return values[start:end]
        return values[start:] + values[:end]


tools = Tools
