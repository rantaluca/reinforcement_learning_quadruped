import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

planeId = p.loadURDF("plane.urdf")

startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])

boxId = p.loadURDF("spot_v1_description/urdf/spot_v1.urdf",startPos, startOrientation) #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId) 
print(cubePos,cubeOrn)
p.disconnect()

