#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import math
import numpy as np
from .framework import (Framework, Keys, main)
from Box2D import (b2Body, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)

frameCounter = 0

with open('TestRun.csv', 'r') as f:
    TestRun = np.loadtxt(f, dtype=np.float64)
    TestRun = TestRun.reshape(-1, 6, 2)

with open('OneFramePrediction.csv', 'r') as f:
    OneFramePrediction = np.loadtxt(f, dtype=np.float64)
    OneFramePrediction = OneFramePrediction.reshape(-1, 6, 2)

with open('TotalPrediction.csv', 'r') as f:
    TotalPrediction = np.loadtxt(f, dtype=np.float64)
    TotalPrediction = TotalPrediction.reshape(-1, 6, 2)


class PoolSim(Framework):

    # Debug framework information
    name = "PoolSim"
    description = "Initial experimentation with pyBox2D"
    bodies = []
    joints = []

    def __init__(self):
        super(PoolSim, self).__init__()

        # Initialize world, no gravity as this is a top down view
        world = self.world
        self.world.gravity = (0, 0)
        ballCount = 6

        # Rectangular boundaries representing pool table
        world.CreateBody(
            shapes=b2LoopShape(vertices=[(0, 0), (0, 40),
                                         (20, 40), (20, 0),
                                         ]),
        )

        world.CreateBody(
            shapes=b2LoopShape(vertices=[(30, 0), (30, 40),
                                         (50, 40), (50, 0),
                                         ]),
        )

        world.CreateBody(
            shapes=b2LoopShape(vertices=[(60, 0), (60, 40),
                                         (80, 40), (80, 0),
                                         ]),
        )

        # Fixture containing the parameters for balls
        ball = b2FixtureDef(
            shape=b2CircleShape(radius=1),
            density=1.0,
            friction=10,
            restitution=0.5,
        )

        # Create balls for testing
        # Damping will be our equivalent of gravity, providing a constant friction-like force

        for i in range(ballCount):
            world.CreateDynamicBody(
                fixtures=ball,
                linearDamping=1,
                angularDamping=2,
                bullet=True,
                # Slight offset to prevent balls spawning outside of the table
                # Balls that spawn to close to each other will correct themselves after 1 frame
                position=(random.uniform(0.5, 19.5), random.uniform(0.5, 39.5)))

        for i in range(ballCount):
            world.CreateDynamicBody(
                fixtures=ball,
                linearDamping=1,
                angularDamping=2,
                bullet=True,
                # Slight offset to prevent balls spawning outside of the table
                # Balls that spawn to close to each other will correct themselves after 1 frame
                position=(random.uniform(30.5, 19.5), random.uniform(30.5, 39.5)))

        for i in range(ballCount):
            world.CreateDynamicBody(
                fixtures=ball,
                linearDamping=1,
                angularDamping=2,
                bullet=True,
                # Slight offset to prevent balls spawning outside of the table
                # Balls that spawn to close to each other will correct themselves after 1 frame
                position=(random.uniform(60.5, 19.5), random.uniform(60.5, 39.5)))

    def Step(self, settings):

        global frameCounter

        for i in range(6):
            self.world.bodies[i+4].position = (TestRun[frameCounter, i, 0], TestRun[frameCounter, i, 1])
            self.world.bodies[i + 10].position = (OneFramePrediction[frameCounter, i, 0] + 30, OneFramePrediction[frameCounter, i, 1])
            self.world.bodies[i + 16].position = (TotalPrediction[frameCounter, i, 0] + 60, TotalPrediction[frameCounter, i, 1])

        frameCounter += 1

        if frameCounter == 3:
            frameCounter = 0

        super(PoolSim, self).Step(settings)


if __name__ == "__main__":
    main(PoolSim)