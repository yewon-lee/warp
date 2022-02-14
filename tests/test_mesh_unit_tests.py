import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np

import test_base

wp.init()


@wp.func
def min_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def max_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.kernel
def compute_bounds(
    indices: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    x0 = positions[i]  # point zero
    x1 = positions[j]  # point one
    x2 = positions[k]  # point two

    lower = min_vec3(min_vec3(x0, x1), x2)
    upper = max_vec3(max_vec3(x0, x1), x2)

    lowers[tid] = lower
    uppers[tid] = upper


@wp.kernel
def compute_num_contacts(
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    counts: wp.array(dtype=int),
):

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    query = wp.mesh_query_aabb(mesh_id, lower, upper)

    index = int(-1)
    count = int(0)

    while wp.mesh_query_aabb_next(query, index):
        count = count + 1

    counts[tid] = count



def test_compute_bounds(test, device):
    
    # create two touching triangles.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, -1, 1]])
    indices = np.array([0, 1, 2, 1, 2, 3])
    m = wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        None,
        wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    # First compute bounds of each of the triangles.
    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    lower_view = lowers.numpy()
    upper_view = uppers.numpy()
    wp.synchronize()

    # Confirm the bounds of each triangle are correct.
    test.assertTrue(lower_view[0][0] == 0)
    test.assertTrue(lower_view[0][1] == 0)
    test.assertTrue(lower_view[0][2] == 0)

    test.assertTrue(upper_view[0][0] == 1)
    test.assertTrue(upper_view[0][1] == 1)
    test.assertTrue(upper_view[0][2] == 0)

    test.assertTrue(lower_view[1][0] == -1)
    test.assertTrue(lower_view[1][1] == -1)
    test.assertTrue(lower_view[1][2] == 0)

    test.assertTrue(upper_view[1][0] == 1)
    test.assertTrue(upper_view[1][1] == 1)
    test.assertTrue(upper_view[1][2] == 1)

def test_mesh_query_aabb_count_overlap(test, device):
    
    # create two touching triangles.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, -1, 1]])
    indices = np.array([0, 1, 2, 1, 2, 3])
    m = wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        None,
        wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    # Compute AABB of each of the triangles.
    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    counts = wp.empty(n=num_tris, dtype=int, device=device)

    wp.launch(
        kernel=compute_num_contacts,
        dim=num_tris,
        inputs=[lowers, uppers, m.id],
        outputs=[counts],
        device=device,
    )

    wp.synchronize()

    view = counts.numpy()

    # 2 triangles that share a vertex having overlapping AABBs.
    for c in view:
        test.assertTrue(c == 2)

def test_mesh_query_aabb_count_nonoverlap(test, device):
    
    # create two separate triangles.
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [10, 1, 0], [10, 0, 1]]
    )
    indices = np.array([0, 1, 2, 3, 4, 5])
    m = wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        None,
        wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    counts = wp.empty(n=num_tris, dtype=int, device=device)

    wp.launch(
        kernel=compute_num_contacts,
        dim=num_tris,
        inputs=[lowers, uppers, m.id],
        outputs=[counts],
        device=device,
    )

    wp.synchronize()

    view = counts.numpy()

    # AABB query only returns one triangle at a time, the triangles are not close enough to overlap.
    for c in view:
        test.assertTrue(c == 1)


devices = wp.get_devices()

class TestMeshQueryAABBMethods(test_base.TestBase):
    pass

TestMeshQueryAABBMethods.add_function_test("test_compute_bounds", test_compute_bounds, devices=devices)
TestMeshQueryAABBMethods.add_function_test("test_mesh_query_aabb_count_overlap", test_mesh_query_aabb_count_overlap, devices=devices)
TestMeshQueryAABBMethods.add_function_test("test_mesh_query_aabb_count_nonoverlap", test_mesh_query_aabb_count_nonoverlap, devices=devices)

if __name__ == '__main__':
    unittest.main(verbosity=2)