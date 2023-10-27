# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Collision test environment
#
###########################################################################

import os
import warp as wp
import warp.sim

from environment import Environment, run_env


class CollisionTestEnvironment(Environment):
    sim_name = "env_collision_test"
    env_offset = (0.5, 0.0, 0.5)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5

    xpbd_settings = dict(iterations=10, enable_restitution=False)

    activate_ground_plane = False
    use_graph_capture = False

    up_axis = "z"
    # gravity = -980.0
    # gravity = -98.0
    gravity = -500.0

    show_rigid_contact_points = True
    contact_points_radius = 1e-1

    rigid_contact_margin = 0.001

    edge_sdf_iter = 100

    num_envs = 1

    show_joints = False

    # rigid_contact_margin = 0.001


    def create_articulation(self, builder):

        # builder.add_shape_plane(plane=(0.0, 0.0, -1.0, 0.0), body=-1, length=10, width=10)
        builder.add_shape_box(body=-1, pos=(0, 0, -5), hx=5, hy=5, hz=5)


        self.rigid_body = builder.add_body(m=0.1, origin=wp.transform(
            ((0.0, 0.0, 1.6)), wp.quat_rpy(0.6, 0.0, 0.4)))
        builder.add_shape_box(
            body=self.rigid_body,
            hx=0.5,
            hy=0.5,
            hz=0.5
        )


if __name__ == "__main__":
    run_env(CollisionTestEnvironment)
