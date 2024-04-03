# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import numpy as np
import torch
from torch import nn

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

num_substeps = 5
SIM_DT = 1 / 60 / num_substeps


class NeuralNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class WarpSimFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, joint_q, joint_qd, joint_act, model, integrator):
        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        ctx.tape = wp.Tape()
        ctx.model = model

        # allocate states for the substeps
        ctx.states = [model.state(requires_grad=True) for _ in range(num_substeps + 1)]

        # ctx.state_in = model.state()
        # ctx.state_out = model.state()

        ctx.states[0].joint_q.assign(wp.from_torch(joint_q))
        ctx.states[0].joint_qd.assign(wp.from_torch(joint_qd))
        ctx.states[0].joint_act.assign(wp.from_torch(joint_act))

        # with ctx.tape:
        #     wp.sim.eval_fk(model, ctx.joint_q, ctx.joint_qd, None, ctx.state)

        with ctx.tape:
            for i in range(num_substeps):
                integrator.simulate(model, ctx.states[i], ctx.states[i + 1], SIM_DT)

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        return (wp.to_torch(ctx.states[-1].joint_q), wp.to_torch(ctx.states[-1].joint_qd))

    @staticmethod
    def backward(ctx, adj_joint_q, adj_joint_qd):
        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        # map incoming Torch grads to our output variables
        ctx.states[-1].joint_q.grad = wp.from_torch(adj_joint_q)
        ctx.states[-1].joint_qd.grad = wp.from_torch(adj_joint_qd)

        ctx.tape.backward()

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        # return adjoint w.r.t. inputs
        return (
            wp.to_torch(ctx.tape.gradients[ctx.states[0].joint_q]),
            wp.to_torch(ctx.tape.gradients[ctx.states[0].joint_qd]),
            wp.to_torch(ctx.tape.gradients[ctx.states[0].joint_act]),
            None,
            None,
        )


class Example:
    frame_dt = 1.0 / 60.0

    episode_duration = 20.0  # seconds
    episode_frames = int(episode_duration / frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)

    sim_time = 0.0

    def __init__(self, stage=None, render=True, num_envs=1):
        builder = wp.sim.ModelBuilder()

        self.enable_rendering = render

        self.num_envs = num_envs

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"),
            articulation_builder,
            xform=wp.transform(np.zeros(3), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
        )

        builder = wp.sim.ModelBuilder()

        for i in range(num_envs):
            builder.add_builder(
                articulation_builder, xform=wp.transform(np.array((i * 2.0, 4.0, 0.0)), wp.quat_identity())
            )

            # joint initial positions
            builder.joint_q[-3:] = [0.0, 0.3, 0.0]

            builder.joint_target[:3] = [0.0, 0.0, 0.0]

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.integrator = wp.sim.FeatherstoneIntegrator()

        # finalize model
        self.model = builder.finalize(requires_grad=True, integrator=self.integrator)  # XXX note the requires_grad=True
        self.model.ground = False

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        # -----------------------
        # set up Usd renderer
        self.renderer = None
        if render:
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=15.0)

    # def update(self):
    #     for _ in range(self.sim_substeps):
    #         self.state.clear_forces()
    #         self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)

    # def render(self, is_live=False):
    #     time = 0.0 if is_live else self.sim_time

    #     self.renderer.begin_frame(time)
    #     self.renderer.render(self.state)
    #     self.renderer.end_frame()

    # def run(self, render=True):
    #     # ---------------
    #     # run simulation

    #     self.sim_time = 0.0
    #     self.state = self.model.state()

    #     wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

    #     profiler = {}

    #     # create update graph
    #     wp.capture_begin()

    #     # simulate
    #     self.update()

    #     graph = wp.capture_end()

    #     # simulate
    #     with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):
    #         for f in range(0, self.episode_frames):
    #             with wp.ScopedTimer("simulate", active=True):
    #                 wp.capture_launch(graph)
    #             self.sim_time += self.frame_dt

    #             if self.enable_rendering:
    #                 with wp.ScopedTimer("render", active=True):
    #                     self.render()
    #                 # self.renderer.save()

    #         wp.synchronize()

    #     avg_time = np.array(profiler["simulate"]).mean() / self.episode_frames
    #     avg_steps_second = 1000.0 * float(self.num_envs) / avg_time

    #     print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

    #     return 1000.0 * float(self.num_envs) / avg_time


model_creator = Example()
model = model_creator.model
state = model.state()

joint_q = wp.to_torch(state.joint_q)
joint_qd = wp.to_torch(state.joint_qd)
joint_act = wp.to_torch(state.joint_act)


torch_device = "cuda"

state_dim = len(joint_q) + len(joint_qd)
residual_model = NeuralNetwork(state_dim).to(torch_device)
print(residual_model)

# X = torch.rand(1, 28, 28, device=torch_device)
# logits = residual_model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# for name, param in residual_model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

simulate = WarpSimFunction.apply

traj_len = 100
small_offset = 0.1234

# generating a ground-truth trajectory
with torch.no_grad():
    reference_traj = torch.empty((traj_len, state_dim), device=torch_device)
    for i in range(100):
        joint_q, joint_qd = simulate(joint_q, joint_qd, joint_act, model, model_creator.integrator)
        reference_traj[i] = torch.cat([joint_q + small_offset, joint_qd])

# plot reference trajectory
import matplotlib.pyplot as plt

plt.plot(reference_traj.cpu().numpy())
plt.show()

# train residual model given reference trajectory
losses = torch.zeros(traj_len, device=torch_device)
rollout_trajectory = torch.empty((traj_len, state_dim), device=torch_device)
for i in range(traj_len):
    joint_q, joint_qd = simulate(joint_q, joint_qd, joint_act, model, model_creator.integrator)
    # apply residual model
    torch_state = torch.empty((1, state_dim), device=torch_device)
    torch_state[0] = torch.cat([joint_q, joint_qd])
    residual = residual_model(torch_state)
    joint_q += residual[0, : len(joint_q)] * 1e-4
    joint_qd += residual[0, len(joint_q) :] * 1e-4

    losses[i] = torch.sum((reference_traj[i] - torch.cat([joint_q, joint_qd])) ** 2)
    rollout_trajectory[i] = torch.cat([joint_q, joint_qd])

# plot rollout trajectory
plt.plot(rollout_trajectory.detach().cpu().numpy())
plt.show()

loss = torch.sum(losses) / traj_len


# here goes SVGD
# gradient residual + Warp simulation
loss.backward()

print(loss)
# print(residual_model.linear_relu_stack[0].weight.grad)
for name, param in residual_model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} | Grads : {param.grad[:2]}\n")

# update SVGD particles using this gradient
act = wp.from_torch(joint_act)
act.grad
