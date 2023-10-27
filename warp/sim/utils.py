import warp as wp
import numpy as np

from typing import Tuple, List

PI = wp.constant(3.14159265359)
PI_2 = wp.constant(1.57079632679)


@wp.func
def velocity_at_point(qd: wp.spatial_vector, r: wp.vec3):
    """
    Returns the velocity of a point relative to the frame with the given spatial velocity.
    """
    return wp.cross(wp.spatial_top(qd), r) + wp.spatial_bottom(qd)


@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    """
    Returns the twist around an axis.
    """

    # project imaginary part onto axis
    a = wp.vec3(q[0], q[1], q[2])
    proj = wp.dot(a, axis)
    a = proj * axis
    # if proj < 0.0:
    #     # ensure twist points in same direction as axis
    #     a = -a
    return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))


@wp.func
def quat_twist_angle(axis: wp.vec3, q: wp.quat):
    """
    Returns the angle of the twist around an axis.
    """
    return 2.0 * wp.acos(quat_twist(axis, q)[3])


@wp.func
def quat_decompose(q: wp.quat):
    """
    Decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x.
    """

    R = wp.mat33(
        wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)),
    )

    # https://www.sedris.org/wg8home/Documents/WG80485.pdf
    phi = wp.atan2(R[1, 2], R[2, 2])
    sinp = -R[0, 2]
    if wp.abs(sinp) >= 1.0:
        theta = 1.57079632679 * wp.sign(sinp)
    else:
        theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])

    return -wp.vec3(phi, theta, psi)


@wp.func
def quat_to_rpy(q: wp.quat):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = wp.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = wp.clamp(t2, -1.0, 1.0)
    pitch_y = wp.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = wp.atan2(t3, t4)

    return wp.vec3(roll_x, pitch_y, yaw_z)


@wp.func
def quat_to_euler(q: wp.quat, i: int, j: int, k: int) -> wp.vec3:
    """
    Convert a quaternion into Euler angles
    i, j, k are the indices in [1,2,3] of the axes to use
    (i != j, j != k)
    """
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276302
    not_proper = True
    if i == k:
        not_proper = False
        k = 6 - i - j  # because i + j + k = 1 + 2 + 3 = 6
    e = float((i - j) * (j - k) * (k - i)) / 2.0  # Levi-Civita symbol
    a = q[0]
    b = q[i]
    c = q[j]
    d = q[k] * e
    if not_proper:
        a -= q[j]
        b += q[k] * e
        c += q[0]
        d -= q[i]
    t2 = wp.acos(2.0 * (a * a + b * b) / (a * a + b * b + c * c + d * d) - 1.0)
    tp = wp.atan2(b, a)
    tm = wp.atan2(d, c)
    t1 = 0.0
    t3 = 0.0
    if wp.abs(t2) < 1e-6:
        t3 = 2.0 * tp - t1
    elif wp.abs(t2 - PI_2) < 1e-6:
        t3 = 2.0 * tm + t1
    else:
        t1 = tp - tm
        t3 = tp + tm
    if not_proper:
        t2 -= PI_2
        t3 *= e
    return wp.vec3(t1, t2, t3)


@wp.func
def quat_from_euler(e: wp.vec3, i: int, j: int, k: int) -> wp.quat:
    """
    Convert Euler angles into a quaternion
    i, j, k are the indices in [1,2,3] of the axes to use
    (i != j, j != k)
    """
    qx = wp.quat(1.0, 0.0, 0.0, e[0])
    qy = wp.quat(0.0, 1.0, 0.0, e[1])
    qz = wp.quat(0.0, 0.0, 1.0, e[2])
    if i == 1:
        qi = qx
    elif i == 2:
        qi = qy
    else:
        qi = qz
    if j == 1:
        qj = qx
    elif j == 2:
        qj = qy
    else:
        qj = qz
    if k == 1:
        qk = qx
    elif k == 2:
        qk = qy
    else:
        qk = qz
    return qi * qj * qk


@wp.func
def quat_between_vectors(a: wp.vec3, b: wp.vec3) -> wp.quat:
    """
    Compute the quaternion that rotates vector a to vector b
    """
    a = wp.normalize(a)
    b = wp.normalize(b)
    c = wp.cross(a, b)
    d = wp.dot(a, b)
    q = wp.quat(c[0], c[1], c[2], 1.0 + d)
    return wp.normalize(q)


@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
    # Frank & Park definition 3.20, pg 100

    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


@wp.func
def transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates).
    (Frank & Park, section 8.2.3, pg 290)
    """

    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.mat33(r1, r2, r3)
    S = wp.mul(wp.skew(p), R)

    T = wp.spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.func
def vec_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def vec_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.func
def vec_abs(a: wp.vec3):
    return wp.vec3(wp.abs(a[0]), wp.abs(a[1]), wp.abs(a[2]))


def load_mesh(filename, use_meshio=True):
    """
    Loads a 3D triangular surface mesh from a file.

    Args:
        filename: The path to the 3D model file (obj, and other formats supported by meshio/openmesh) to load.
        use_meshio: If True, use meshio to load the mesh. Otherwise, use openmesh.

    Returns:
        Tuple of (mesh_points, mesh_indices), where mesh_points is a Nx3 numpy array of vertex positions (float32),
        and mesh_indices is a Mx3 numpy array of vertex indices (int32) for the triangular faces.
    """
    if use_meshio:
        import meshio
        m = meshio.read(filename)
        mesh_points = np.array(m.points)
        mesh_indices = np.array(m.cells[0].data, dtype=np.int32)
    else:
        import openmesh
        m = openmesh.read_trimesh(filename)
        mesh_points = np.array(m.points())
        mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32)
    return mesh_points, mesh_indices


def visualize_meshes(meshes: List[Tuple[list, list]], num_cols=0, num_rows=0, titles=[], scale_axes=True, show_plot=True):
    # render meshes in a grid with matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    num_cols = min(num_cols, len(meshes))
    num_rows = min(num_rows, len(meshes))
    if num_cols and not num_rows:
        num_rows = int(np.ceil(len(meshes) / num_cols))
    elif num_rows and not num_cols:
        num_cols = int(np.ceil(len(meshes) / num_rows))
    else:
        num_cols = len(meshes)
        num_rows = 1

    vertices = [np.array(v).reshape((-1, 3)) for v, _ in meshes]
    faces = [np.array(f, dtype=np.int32).reshape((-1, 3)) for _, f in meshes]
    if scale_axes:
        ranges = np.array([v.max(axis=0) - v.min(axis=0) for v in vertices])
        max_range = ranges.max()
        mid_points = np.array([v.max(axis=0) + v.min(axis=0) for v in vertices]) * 0.5

    fig = plt.figure(figsize=(12, 6))
    for i, (vertices, faces) in enumerate(meshes):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')
        if i < len(titles):
            ax.set_title(titles[i])
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, edgecolor='k')
        if scale_axes:
            mid = mid_points[i]
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    if show_plot:
        plt.show()
    return fig


def remesh_ftetwild(vertices, faces, stop_quality=10, max_its=50, edge_length_r=0.1, epsilon=0.01):
    """
    Remeshes a 3D triangular surface mesh using "Fast Tetrahedral Meshing in the Wild" (fTetWild).
    This is useful for improving the quality of the mesh, and for ensuring that the mesh is
    watertight. This function first tetrahedralizes the mesh, then extracts the surface mesh.
    The resulting mesh is guaranteed to be watertight and may have a different topology than the
    input mesh.

    This function requires that wildmeshing is installed, see
    https://wildmeshing.github.io/python/ for installation instructions.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        stop_quality: The maximum AMIPS energy for stopping mesh optimization.
        max_its: The maximum number of mesh optimization iterations.
        edge_length_r: The relative target edge length as a fraction of the bounding box diagonal.
        epsilon: The relative envelope size as a fraction of the bounding box diagonal.
        visualize: If True, visualize the input mesh next to the remeshed result using matplotlib.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh. Returns the original vertices and faces
        if the remeshing fails.
    """
    import wildmeshing as wm
    from collections import defaultdict

    tetra = wm.Tetrahedralizer(
        stop_quality=stop_quality, max_its=max_its, edge_length_r=edge_length_r, epsilon=epsilon)
    tetra.set_mesh(vertices, np.array(faces).reshape(-1, 3))
    tetra.tetrahedralize()
    tet_vertices, tet_indices, _ = tetra.get_tet_mesh()

    def face_indices(tet):
        face1 = (tet[0], tet[2], tet[1])
        face2 = (tet[1], tet[2], tet[3])
        face3 = (tet[0], tet[1], tet[3])
        face4 = (tet[0], tet[3], tet[2])
        return (
            (face1, tuple(sorted(face1))),
            (face2, tuple(sorted(face2))),
            (face3, tuple(sorted(face3))),
            (face4, tuple(sorted(face4))))

    # determine surface faces
    elements_per_face = defaultdict(set)
    unique_faces = {}
    for e, tet in enumerate(tet_indices):
        for face, key in face_indices(tet):
            elements_per_face[key].add(e)
            unique_faces[key] = face
    surface_faces = [face for key, face in unique_faces.items() if len(elements_per_face[key]) == 1]

    new_vertices = np.array(tet_vertices)
    new_faces = np.array(surface_faces, dtype=np.int32)

    if len(new_vertices) == 0 or len(new_faces) == 0:
        import warnings
        warnings.warn("Remeshing failed, the optimized mesh has no vertices or faces; return previous mesh.")
        return vertices, faces

    return new_vertices, new_faces


def remesh_alphashape(vertices, faces=None, alpha=3.0):
    """
    Remeshes a 3D triangular surface mesh using the alpha shape algorithm.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces (not needed).
        alpha: The alpha shape parameter.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh.
    """
    import alphashape

    alpha_shape = alphashape.alphashape(vertices, alpha)
    return np.array(alpha_shape.vertices), np.array(alpha_shape.faces, dtype=np.int32)


def remesh(vertices, faces, method="ftetwild", visualize=False, **remeshing_kwargs):
    """
    Remeshes a 3D triangular surface mesh using the specified method.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        method: The remeshing method to use. One of "ftetwild" or "alphashape".
        visualize: Whether to render the input and output meshes using matplotlib.
        **remeshing_kwargs: Additional keyword arguments passed to the remeshing function.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh.
    """
    if method == "ftetwild":
        new_vertices, new_faces = remesh_ftetwild(vertices, faces, **remeshing_kwargs)
    elif method == "alphashape":
        new_vertices, new_faces = remesh_alphashape(vertices, faces, **remeshing_kwargs)
    # TODO add poisson sampling (trimesh has implementation at https://trimsh.org/trimesh.sample.html)
    else:
        raise ValueError(f"Unknown remeshing method: {method}")

    if visualize:
        # side-by-side visualization of the input and output meshes
        visualize_meshes([(vertices, faces), (new_vertices, new_faces)], titles=["Original", "Remeshed"])
    return new_vertices, new_faces


def plot_graph(vertices, edges, edge_labels=[]):
    """
    Plots a graph using matplotlib.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        edges: A numpy array of shape (M, 2) containing the vertex indices of the edges.
        edge_labels: A list of edge labels.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.DiGraph()
    name_to_index = {}
    for i, name in enumerate(vertices):
        G.add_node(i)
        name_to_index[name] = i
    g_edge_labels = {}
    for i, (a, b) in enumerate(edges):
        a = a if isinstance(a, int) else name_to_index[a]
        b = b if isinstance(b, int) else name_to_index[b]
        label = None
        if i < len(edge_labels):
            label = edge_labels[i]
            g_edge_labels[(a, b)] = label
        G.add_edge(a, b, label=label)

    # try:
    #     pos = nx.nx_agraph.graphviz_layout(
    #         G, prog='neato', args='-Gnodesep="10" -Granksep="10"')
    # except:
    #     print(
    #         "Warning: could not use graphviz to layout graph. Falling back to spring layout.")
    #     print("To get better layouts, install graphviz and pygraphviz.")
    #     pos = nx.spring_layout(G, k=3.5, iterations=200)
    #     # pos = nx.kamada_kawai_layout(G, scale=1.5)
    #     # pos = nx.spectral_layout(G, scale=1.5)
    pos = nx.nx_agraph.graphviz_layout(
        G, prog='neato', args='-Gnodesep="20" -Granksep="20"')

    default_draw_args = dict(
        alpha=0.9, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_nodes(G, pos, **default_draw_args)
    nx.draw_networkx_labels(G, pos, labels={i: v for i, v in enumerate(vertices)}, font_size=8, bbox=dict(
        facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True, edge_color='black', node_size=1000)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=g_edge_labels,
        font_color='darkslategray',
        font_size=8,
    )
    plt.axis('off')
    plt.show()
