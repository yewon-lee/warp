# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Spray bottle environment
#
# Shows how to load a URDF of a spray bottle.
#
###########################################################################

import os
import warp as wp
import warp.sim

from environment import Environment, run_env

class CartpoleEnvironment(Environment):
    sim_name = "env_cartpole"
    env_offset=(2.0, 0.0, 2.0)
    nano_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 5
    
    activate_ground_plane = False

    num_envs = 9

    def create_articulation(self, builder):
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "../assets/spray_bottle/mobility.urdf"),
            builder,
            xform=wp.transform(),
            floating=False, 
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False)


if __name__ == "__main__":
    run_env(CartpoleEnvironment)
