# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Throw Rigid
#
# Optimize the initial velocity of a rigid body to hit a target.
#
###########################################################################

from warp.tests.grad_utils import *
from warp.optim import Adam, SGD
import warp.sim.render
import warp.sim
from warp.sim import ModelBuilder
import warp as wp
import numpy as np
import os
DEBUG = False


if DEBUG:
    wp.config.verify_cuda = True
    wp.config.verify_fp = True
    wp.config.mode = "debug"


wp.init()
# if DEBUG:
#     wp.set_device("cpu")


@wp.kernel
def sim_loss(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector), target_pos: wp.vec3, loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    tf = body_q[i]
    dist = wp.length_sq(wp.transform_get_translation(tf) - target_pos)
    vel = wp.length_sq(body_qd[i])
    l = dist + 0.1 * vel
    loss[0] = l


@wp.kernel
def apply_velocity(action: wp.array(dtype=wp.vec3), body_qd: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    # apply linear velocity
    body_qd[i] = wp.spatial_vector(wp.vec3(0.0), action[i])


class Environment:
    frame_dt = 1.0 / 60.0
    episode_frames = 100

    sim_substeps = 5
    sim_dt = frame_dt / sim_substeps

    sim_time = 0.0
    render_time = 0.0

    def __init__(self, device="cpu"):
        self.device = device

        self.start_pos = wp.vec3(0.0, 1.6, 0.0)
        self.target_pos = wp.vec3(3.0, 0.6, 0.0)
        ModelBuilder.default_shape_kd = 10.0
        ModelBuilder.default_shape_ke = 1e6

        builder = ModelBuilder()
        builder.add_articulation()
        b = builder.add_body(origin=wp.transform(self.start_pos))
        _ = builder.add_shape_box(pos=(0.0, 0.0, 0.0), hx=0.5, hy=0.5, hz=0.5, density=1000.0, ke=1e7, kd=1e5, body=b)
        # _ = builder.add_shape_sphere(pos=(0.0, 0.0, 0.0), radius=0.5, density=1000.0, body=b, thickness=1e-2)

        solve_iterations = 2
        self.integrator = wp.sim.XPBDIntegrator(solve_iterations)
        # self.integrator = wp.sim.SemiImplicitIntegrator()

        # finalize model
        self.model = builder.finalize(device, requires_grad=True, integrator=self.integrator)

        self.builder = builder
        self.model.ground = True

        self.num_iterations = 100

        self.capture_graph = not DEBUG

    def simulate(self) -> wp.sim.State:
        """
        Simulate the system for the given states.
        """

        self.render_time = 0.0
        traj_verts = []
        for frame in range(self.episode_frames):
            for i in range(self.sim_substeps):
                t = frame * self.sim_substeps + i
                state = self.states[t]
                next_state = self.states[t + 1]
                state.clear_forces()

                wp.sim.collide(self.model, state)
                self.integrator.simulate(self.model, state, next_state, self.sim_dt)

            if self.renderer is not None:
                with wp.ScopedTimer("render", active=False):
                    self.renderer.render_sphere("target", self.target_pos, wp.quat_identity(), 0.1)
                    self.renderer.begin_frame(self.render_time)
                    self.renderer.render(state)
                    self.renderer.end_frame()
                    self.render_time += self.frame_dt
                    traj_verts.append(next_state.body_q.numpy()[0, :3].tolist())
                    self.renderer.render_line_strip(
                        vertices=traj_verts,
                        color=wp.render.bourke_color_map(0.0, self.num_iterations, self.iteration),
                        radius=0.02 + 0.01 * self.iteration / self.num_iterations,
                        name=f"traj_{self.iteration}",
                    )

    def dynamics(self, action):
        # apply initial velocity to the rigid object
        wp.launch(apply_velocity, 1, inputs=[action], outputs=[self.states[0].body_qd], device=action.device)

        self.simulate()
        final_state = self.states[-1]

        wp.launch(sim_loss, dim=1, inputs=[final_state.body_q, final_state.body_qd,
                  self.target_pos], outputs=[self.loss], device=action.device)

        return self.loss

    def optimize(self, num_iter=100, lr=0.01, render=True):
        action = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=self.device)

        optimizer = Adam([action], lr=lr)
        # optimizer = SGD([action], lr=lr, nesterov=True, momentum=0.1)

        self.num_iterations = num_iter
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=self.device)

        tape_buffer = wp.zeros(10000, dtype=wp.int32, device=self.device)

        with wp.ScopedTimer("allocate states"):
            self.states = [self.model.state() for _ in range(self.episode_frames * self.sim_substeps + 1)]

        if render:
            # set up Usd renderer
            self.renderer = wp.sim.render.SimRendererOpenGL(
                self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_trajopt.usd"), scaling=1.0
            )
            if self.capture_graph:
                print("WARNING: capture_graph is not supported with render=True, setting to False")
            self.capture_graph = False
        else:
            self.renderer = None

        if self.capture_graph:
            wp.capture_begin()
            tape = wp.Tape(buffer=tape_buffer)
            with tape:
                self.dynamics(action)
            tape.backward(loss=self.loss)
            graph = wp.capture_end()

        # check_tape_safety(self.dynamics, [action])

        # check_backward_pass(
        #     tape,
        #     visualize_graph=False,
        #     analyze_graph=False,
        #     track_inputs=[action],
        #     track_outputs=[self.loss],
        #     track_input_names=["action"],
        #     track_output_names=["loss"])

        # optimize
        losses = []
        for i in range(1, num_iter + 1):
            self.iteration = i

            for i, state in enumerate(self.states):
                for key, value in state.__dict__.items():
                    if isinstance(value, wp.array):
                        if len(value) == 0 or not value.grad:
                            continue
                        value.grad.zero_()

            if self.capture_graph:
                wp.capture_launch(graph)
            else:
                tape = wp.Tape(buffer=tape_buffer)
                with tape:
                    self.dynamics(action)
                tape.backward(loss=self.loss)

                if False:
                    check_tape_safety(self.dynamics, [action])

                if False:
                    array_names = {}
                    populate_array_names(self.model, array_names, prefix="model.")
                    for state_id, state in enumerate(self.states):
                        populate_array_names(state, array_names, prefix=f"state_{state_id}.")

                    check_backward_pass(
                        tape,
                        render_mermaid=os.path.join(os.path.dirname(__file__), "example_sim_throw_rigid.html"),
                        # render_d2=os.path.join(os.path.dirname(__file__), "example_sim_throw_rigid.d2"),
                        analyze_graph=False,
                        # check_kernel_jacobians=False,
                        # simplify_graph=False,  # we want to see all input/output nodes
                        track_inputs=[action],
                        track_outputs=[self.loss],
                        track_input_names=["action"],
                        track_output_names=["loss"],
                        array_names=array_names)

                if True:
                    plot_state_gradients(self.states, os.path.join(os.path.dirname(
                        __file__), "example_sim_throw_rigid_state_grads.html"))

            if False:
                check_jacobian(
                    self.dynamics,
                    inputs=[action],
                    input_names=["action"],
                    output_names=["loss"],
                    jacobian_name="throw_rigid_loss",
                )

            l = self.loss.numpy()[0]
            print(f"iter {i}/{num_iter}\t action: {action.numpy()}\t action.grad: {action.grad.numpy()}\t loss: {l:.3f}")
            losses.append(l)

            # print("action grad", opt_vars.grad.numpy())
            assert not np.isnan(action.grad.numpy()).any(), "Gradient contains NaN"
            optimizer.step([action.grad])
            tape.zero()

        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.grid()
        plt.title("Loss")
        plt.xlabel("Iteration")
        plt.show()

        return action


np.set_printoptions(precision=4, linewidth=2000, suppress=True)

if DEBUG:
    sim = Environment(device="cpu")
else:
    sim = Environment(device=wp.get_preferred_device())

best_actions = sim.optimize(num_iter=80, lr=0.5, render=True)

sim.renderer = wp.sim.render.SimRendererOpenGL(
    sim.model,
    os.path.join(os.path.dirname(__file__), "outputs", "example_sim_trajopt.usd"),
    scaling=1.0)
sim.simulate()
sim.renderer.save()
