#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import math
import numpy as np
from .framework import (Framework, Keys, main)
from Box2D import (b2Body, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)

reset = True
dataTensor = np.array([], dtype=np.float64)
simulationCounter = 0


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

    def Step(self, settings):

        # Track if the simulation has reset
        global reset

        # Stores all frame data for output
        global dataTensor

        # Tracks iteration of the simulator
        global simulationCounter

        # Track if balls are still in motion
        movement = False

        # Apply random impulse to ball at the start of each iteration
        if reset:
            randForce = (random.randint(-1000, 1000), random.randint(-1000, 1000))
            print(randForce)
            self.world.bodies[2].ApplyLinearImpulse(randForce, self.world.bodies[2].worldCenter, True)
            reset = False

        frame = np.array([], dtype=np.float64)

        # If any of the balls are in motion set movement to true
        for body in self.world.bodies:
            # Walls are always awake so we exclude them by position
            # Balls will never be at position 0,0
            if body.position != (0, 0):
                if body.awake:
                    movement = True
                # Records position of each ball
                frame = np.append(frame, np.array([body.position.x, body.position.y]), axis=0)

        # Records each frame
        dataTensor = np.append(dataTensor, frame, axis=0)

        # Once all balls have stopped moving exit the program
        if not movement:
            # Reshape frame data
            # 2 dimensions for the x and y coordinates of each ball
            # 6 dimensions for the number of balls in each frame
            # -1 for the unknown quantity of frames in a single iteration
            dataTensor = dataTensor.reshape(-1, 6, 2)
            # Save data to a numpy file for training the neural net
            f = open('training_data\simulation'+str(simulationCounter)+'.npy', "w")
            np.save(f, dataTensor)
            g = open('training_data\simulation'+str(simulationCounter)+'.csv', "w")
            np.savetxt(g, dataTensor.flatten())
            # Save data into a human readable format
            with file('training_data\simulation'+str(simulationCounter)+'.txt', 'w') as outfile:
                for framenumber, data_slice in enumerate(dataTensor, 1):
                    np.savetxt(outfile, data_slice, fmt='%-10.4f', header='x        y')
                    outfile.write('\n# Frame'+str(framenumber)+'\n\n')
            # Clear our tensor
            dataTensor = np.array([])

            # Reset positions of balls for new simulation
            for body in self.world.bodies:
                if body.position != (0, 0):
                    body.position = (random.uniform(0.5, 19.5), random.uniform(0.5, 39.5))

            # Indicates simulation has reset
            reset = True

            # Increment simulation counter for file names
            simulationCounter += 1

        super(PoolSim, self).Step(settings)


if __name__ == "__main__":
    main(PoolSim)
