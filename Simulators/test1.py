#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)


class Test1(Framework):

    # Debug framework information
    name = "Test1"
    description = "Initial experimentation with pyBox2D"
    bodies = []
    joints = []
    global constFrameCounter
    constFrameCounter = 0

    def __init__(self):
        super(Test1, self).__init__()

        # Initialize world, no gravity as this is a top down view
        world = self.world
        self.world.gravity = (0, 0)

        # Rectangular boundaries representing pool table
        table = world.CreateBody(
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

        # Create two balls for testing
        # Damping will be our equivalent of gravity, providing a constant friction-like force
        ball1 = world.CreateDynamicBody(
            fixtures=ball,
            linearDamping=1,
            angularDamping=2,
            bullet=True,
            position=(10, 10))

        ball2 = world.CreateDynamicBody(
            fixtures=ball,
            linearDamping=1,
            angularDamping=2,
            bullet=True,
            position=(10, 30))

    def Step(self, settings):

        global constFrameCounter

        # Prints position and angle of all objects every 60 frames
        if constFrameCounter % 60 == 0:
            for body in self.world.bodies:
                print(body.position, body.angle)

        # Increment frame counter 
        constFrameCounter += 1

        super(Test1, self).Step(settings)


if __name__ == "__main__":
    main(Test1)
