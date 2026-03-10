import time

import pybullet as p
import pybullet_data

PHYSICS_FREQ_DT = 1.0 / 240
CONTROL_FREQ_DT = 1.0 / 120

BODY_HALF_EXTENTS = [0.5, 0.5, 0.5]
BODY_MASS = 1

DAMP_LIN = 1e-5
DAMP_ANG = 1E-5

START_POS = [0, 0, 0]
START_ORIENT = p.getQuaternionFromEuler([0, 0, 0])

p.connect(p.GUI)  # use p.GUI for graphics version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(PHYSICS_FREQ_DT)

p.setGravity(0, 0, 0)  # simulates space environment

# setup the objects

cuid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
vuid = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
body = p.createMultiBody(baseMass=BODY_MASS,
                         baseCollisionShapeIndex=cuid,
                         baseVisualShapeIndex=vuid,
                         basePosition=START_POS,
                         baseOrientation=START_ORIENT)

# for numerical stability of dynamics
p.changeDynamics(body, -1,
                 linearDamping=DAMP_LIN,
                 angularDamping=DAMP_ANG)


TARGET_ANG =

while True:
    time.sleep(0.01)

    for _ in range(int(CONTROL_FREQ_DT / PHYSICS_FREQ_DT)):
        p.stepSimulation()
