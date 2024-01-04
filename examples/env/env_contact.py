# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Contact environment
#
# Shows how to set up a simulation of a rigid-body Hopper articulation based on
# the OpenAI gym environment using the Environment class and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)



class ContactEnvironment(Environment):
    sim_name = "env_contact"
    env_offset = (10.0, 0.0, 10.0)
    opengl_render_settings = dict(scaling=1.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 16
    sim_substeps_xpbd = 5

    integrator_type = IntegratorType.FEATHERSTONE
    # integrator_type = IntegratorType.EULER

    xpbd_settings = dict(iterations=7)

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    use_graph_capture = True
    use_tiled_rendering = False
    show_joints = True

    activate_ground_plane = True
    
    use_graph_capture = False
    
    num_envs = 4

    def create_articulation(self, builder):
        shape_ke = 1.0e4
        shape_kd = 1.0e2
        shape_kf = 1.0e4

        builder.set_ground_plane(
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )

        b = builder.add_body()
        # tf = wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.25))
        # tf = wp.transform((0.1, 1.5, 6.0), wp.quat_identity())
        # builder.add_joint_free(
        #     b, tf
        # )
        tf = wp.transform((0.0, 1.5, 0.0), wp.quat_from_axis_angle((0.0, 0.0, 1.0), np.pi * 0.25))
        # builder.add_joint_free(b, child_xform=wp.transform_inverse(tf))
        builder.add_joint_d6(
            parent=-1,
            child=b,
            child_xform=wp.transform_inverse(tf),
            linear_axes=[
                wp.sim.JointAxis((1.0, 0.0, 0.0)),
                wp.sim.JointAxis((0.0, 1.0, 0.0)),
                # wp.sim.JointAxis((0.0, 0.0, 1.0)),
            ],
            angular_axes=[
                # wp.sim.JointAxis((1.0, 0.0, 0.0)),
                wp.sim.JointAxis((0.0, 1.0, 0.0)),
                wp.sim.JointAxis((0.0, 0.0, 1.0)),
            ],
            )
        # builder.joint_q[-7:-4] = tf.p
        # builder.joint_q[-4:] = tf.q

        builder.add_shape_capsule(
            pos=(0.0, 0.0, 0.0),
            radius=0.15,
            half_height=0.3,
            up_axis=0,
            body=b,
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )

        # builder.add_shape_sphere(
        #     pos=(0.0, 0.0, 0.0),
        #     radius=0.15,
        #     body=b,
        #     ke=shape_ke,
        #     kd=shape_kd,
        #     kf=shape_kf,
        # )


if __name__ == "__main__":
    run_env(ContactEnvironment)
