import math
import warp as wp
import warp.sim
import warp.sim.render
import numpy as np
from pxr import Sdf, Gf


class PhysicsSim:
    def __init__(self):

        self.sim_fps = 60.0
        self.sim_substeps = 32
        self.sim_duration = 5.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps

        self.rigid_body = None
        self.finalized = False
        self.builder = wp.sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))

        self.add_rigid_body((0.0, 0.0, 200.0))
        self.add_shape_box()

    def add_rigid_body(self, position):
        self.rigid_body = self.builder.add_body(m=0.1, origin=wp.transform(
            (position[0], position[1], position[2]), wp.quat_identity()))
        self.builder.add_shape_box(
            body=self.rigid_body,
            hx=0.5,
            hy=0.5,
            hz=0.5
        )

    def add_shape_box(self):
        # self.builder.add_shape_plane(plane=(0.0, 0.0, -1.0, 0.0), body=-1, length=10, width=10)

        # Why does this not work?
        self.builder.add_shape_box(body=-1, pos=(0, 0, -5), hx=5, hy=5, hz=5)

    def finalize(self):
        self.model = self.builder.finalize()

        self.integrator = wp.sim.XPBDIntegrator(iterations=10, enable_restitution=True)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.model.up_axis = 2
        self.model.ground = False
        self.model.gravity = (0, 0, -980)
        self.model.up_vector = [0.0, 0.0, 1.0]

        # self.renderer = wp.sim.render.SimRendererOpenGL(
        #     self.model, "Physics sim",
        #     up_axis="z")
        self.renderer = wp.sim.render.SimRendererUsd(
            self.model, "physics_sim.usda",
            up_axis="z")
        # self.renderer.paused = True

    def update(self, sim_time: float):
        if self.finalized == False:
            self.finalize()
            self.finalized = True

        for s in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        with wp.ScopedTimer("render", False):
            self.renderer.begin_frame(sim_time)
            # render state 1 (swapped with state 0 just before)
            self.renderer.render(self.state_1)
            self.renderer.end_frame()


wp.init()
physics_sim = PhysicsSim()

for t in range(physics_sim.sim_frames):
    physics_sim.update(t / physics_sim.sim_fps)

physics_sim.renderer.save()


# def add_shape_box(primPath: Sdf.Path, position: Gf.Vec3f, size: Gf.Vec3f):
#     global physics_sim
#     physics_sim.add_shape_box(primPath, position, size)


# def add_rigid_body(primPath: Sdf.Path, position: Gf.Vec3f, size: Gf.Vec3f):
#     global physics_sim
#     physics_sim.add_rigid_body(primPath, position, size)


# def exec_sim():
#     global physics_sim

#     physics_sim.update(0.0)

#     transform = wp.transform_expand(physics_sim.state_0.body_q.numpy()[-1])
#     m = Gf.Matrix4d().SetIdentity()
#     m.SetTranslate(transform.p)
#     return m
