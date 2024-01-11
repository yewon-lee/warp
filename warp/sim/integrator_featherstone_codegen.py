import warp as wp

def jcalc(type, q, qd):
    pass


# Frank & Park definition 3.20, pg 100
@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
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


def fd(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_inertia: wp.array(dtype=wp.spatial_matrix),
    body_f_ext: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_tau: wp.array(dtype=float),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_mode: wp.array(dtype=wp.int32),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    # outputs
    joint_qdd: wp.array(dtype=float),
):
    # Articulated Body Algorithm, Table 7.1 from Featherstone
    v0 = wp.spatial_vector(0.0)
    for i in range(num_bodies):
        parent = joint_parent[i]
        child = joint_child[i]
        lin_axis_count = joint_axis_dim[i, 0]
        ang_axis_count = joint_axis_dim[i, 1]
        X_j, S_j, v_j = jcalc(
            joint_type[i],
            joint_q,
            joint_qd,
            joint_q_start[i],
            joint_qd_start[i],
            joint_axis_start[i],
            lin_axis_count,
            ang_axis_count,
        )
        S[i] = S_j
        X_p = joint_X_p[i]
        X_c = joint_X_c[i]
        X_pc_i = X_p * X_j * wp.transform_inverse(X_c)
        X_pc[i] = X_pc_i
        X_wc = X_pc_i
        if parent >= 0:
            X_wc = body_q[parent] * X_pc_i
            body_q[child] = X_wc
        v_i = transform_twist(X_pc_i, body_qd[parent]) + v_j
        v[i] = v_i
        c[i] = wp.cross(v_i, v_j)
        I_i = body_inertia[child]
        I_a[i] = I_i
        # v x* I v - X_0 * f_ext
        p_a[i] = spatial_inertia_twist(I_i, v_i) - transform_wrench(X_wc, body_f_ext[child])
    
    for i in range(num_bodies):
        parent = joint_parent[i]
        child = joint_child[i]
        qd_start = joint_qd_start[i]
        S_i = S[i]
        U_i = I_a[i] * S_i
        D_i = I_a[i] * U_i
        u_i = joint_tau[qd_start] - S_i @ p_a[i]
        U[i] = U_i
        D[i] = D_i
        u[i] = u_i
        if parent >= 0:
            D_inv_i = wp.inverse(D_i)
            D_inv[i] = D_inv_i
            X_p = joint_X_p[i] * X_j
            temp_I_a = I_a[i] - U_i @ D_inv_i @ wp.transpose(U_i)
            temp_p_a = p_a[i] + U_i @ D_inv_i @ u_i
            I_a[parent] += transform_inertia(X_p, temp_I_a)
            p_a[parent] += transform_wrench(X_p, temp_p_a)
    
    a[0] = wp.spatial_vector(0.0, 0.0, 0.0, -gravity[0], -gravity[1], -gravity[2])
    for i in range(num_bodies):
        parent = joint_parent[i]
        child = joint_child[i]
        temp_a = transform_twist(X_pc[i], a[parent]) + c[i]
        qdd_i = D_inv[i] @ (u[i] - U[i] @ temp_a)
        joint_qdd[i] = qdd_i
        a[child] = temp_a + S[i] @ qdd_i

