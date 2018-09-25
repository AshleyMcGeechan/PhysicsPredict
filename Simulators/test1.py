#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)


class Test1(Framework):
    name = "Test1"
    description = "Initial experimentation with pyBox2D"
    bodies = []
    joints = []

    def __init__(self):
        super(Test1, self).__init__()

        self.world.gravity = (0, 0)

        table = self.world.CreateBody(
            shapes=b2LoopShape(vertices=[(0, 0), (0, 40),
                                         (20, 40), (20, 0),
                                         ]),
        )

        self.ball = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=1),
                density=1.0,
                friction=20,
                restitution=0.5),
            linearDamping=1,
            angularDamping=2,
            bullet=True,
            position=(10, 10))

        self.ball = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=1),
                density=1.0,
                friction=20,
                restitution=0.5),
            linearDamping=1,
            angularDamping=2,
            bullet=True,
            position=(10, 30))


if __name__ == "__main__":
    main(Test1)
