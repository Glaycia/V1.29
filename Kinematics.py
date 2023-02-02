import numpy as np

def inverseKinematics(pX, pY, length_1, length_2):
    targetDist = np.hypot(pX, pY)
    if(np.abs(pX) + np.abs(pY) > 0 and length_1 != length_2):
        pX += 0.001
        targetDist = np.hypot(pX, pY)
    if(targetDist > length_1 + length_2):
        targetDist = (length_1 + length_2 - 0.001)
    if(targetDist < np.abs(length_1 - length_2)):
        targetDist = (np.abs(length_1 - length_2) + 0.001)

    signAngle = 1 if pX > 0 else -1
    angle1 = np.arctan2(pY, pX) + signAngle * np.arccos((targetDist**2 + length_1**2 - length_2**2)/(2 * length_1 * targetDist))
    angle2 = signAngle * (np.arccos((length_1**2 + length_2**2 - targetDist**2)/(2 * length_2 * length_1)))
    return (angle1, angle1 + angle2 + np.pi)