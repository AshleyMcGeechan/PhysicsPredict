#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import math
from .framework import (Framework, Keys, main)
from Box2D import (b2Body, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)


class PoolSim(Framework):

    # Debug framework information
    name = "PoolSim"
    description = "Initial experimentation with pyBox2D"
    bodies = []
    joints = []

    # Frame counter didn't want to work if it wasn't global and initialised
    # Initialised to 1 so that the movement check doesn't occur before in the second before the impulse is applied
    # I imagine an inbuilt frame counter exists but I haven't found it yet
    global constFrameCounter
    constFrameCounter = 1

    def __init__(self):
        super(PoolSim, self).__init__()

        # Initialize world, no gravity as this is a top down view
        world = self.world
        self.world.gravity = (0, 0)
        ballCount = 4

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

        global constFrameCounter

        # Prints position and angle of all objects every 60 frames
        if constFrameCounter % 60 == 0:
            # Track if balls are still in motion
            movement = False

            # Applies random impulse to one of the balls after 1 second
            if constFrameCounter == 60:
                randForce = (random.randint(-1000, 1000), random.randint(-1000, 1000))
                print(randForce)
                self.world.bodies[2].ApplyLinearImpulse(randForce, self.world.bodies[2].worldCenter, True)

            # If any of the balls are in motion set movement to true
            for body in self.world.bodies:
                # Walls are always awake so we exclude them by position
                # Balls will never be at position 0,0
                if body.awake and body.position != (0, 0):
                    movement = True
                # Print statistics for data gathering
                print(body.position, body.angle, body.awake)

            # Once all balls have stopped moving exit the program
            if not movement:
                exit()

        # Increment frame counter
        constFrameCounter += 1

        super(PoolSim, self).Step(settings)


if __name__ == "__main__":
    main(PoolSim)
