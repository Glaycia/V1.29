import numpy as np

def forwardKinematics(q, length_1, length_2):
    q1 = q[0, 0]
    q2 = q[1, 0]
    x = np.empty((2, 1))
    x[0, 0] = np.cos(q1)*length_1+np.cos(q2)*length_2
    x[1, 0] = np.sin(q1)*length_1+np.sin(q2)*length_2
    return x
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

def inverseVelocities(q1, q2, vX, vY, length_1, length_2):
    #dxi/dqi
    epsilon = 1/(2.0 ** 10)
    q = np.empty((2, 1))
    q[0, 0] = q1
    q[1, 0] = q2
    J = np.empty((2, 2))
    for i in range(2):
        right = q.copy()
        left = q.copy()
        right[i] = right[i] + epsilon
        left[i] = left[i] - epsilon
        diff = ((forwardKinematics(right, length_1, length_2) - forwardKinematics(left, length_1, length_2))/(2 * epsilon))[:, 0]
        for j in range(2):
            J[j, i] = diff[j]
    xdot = np.empty((2, 1))
    xdot[0, 0] = vX
    xdot[1, 0] = vY
    return np.linalg.inv(J) @ xdot

def rkdp(f, x, u, dt, max_error = 1e-6):
    A = np.zeros((6,6))

    A[0,:1] = np.array([1.0 / 5.0])
    A[1,:2] = np.array([3.0 / 40.0, 9.0 / 40.0])
    A[2,:3] = np.array([44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0])
    A[3,:4] = np.array([19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0])
    A[4,:5] = np.array([9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0])
    A[5,:6] = np.array([35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0])

    b1 = np.array([35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0,  0.0])
    b2 = np.array([5179.0 / 57600.0, 0.0, 7571.0 / 16695.0,    393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0])

    time_elapsed = 0.0
    h = dt

    while time_elapsed < dt:
        keep_going = True
        while keep_going:
            h = min(h, dt - time_elapsed)

            k1 = f(x, u)
            k2 = f(x + h * (A[0,0] * k1), u)
            k3 = f(x + h * (A[1,0] * k1 + A[1,1] * k2), u)
            k4 = f(x + h * (A[2,0] * k1 + A[2,1] * k2 + A[2,2] * k3), u)
            k5 = f(x + h * (A[3,0] * k1 + A[3,1] * k2 + A[3,2] * k3 + A[3,3] * k4), u)
            k6 = f(x + h * (A[4,0] * k1 + A[4,1] * k2 + A[4,2] * k3 + A[4,3] * k4 + A[4,4] * k5), u)

            new_x = x + h * (A[5,0] * k1 + A[5,1] * k2 + A[5,2] * k3 + A[5,3] * k4 + A[5,4] * k5 + A[5,5] * k6)
            k7 = f(new_x, u)

            truncation_error = np.linalg.norm(h * ((b1[0] - b2[0]) * k1 + (b1[1] - b2[1]) * k2 +
                              (b1[2] - b2[2]) * k3 + (b1[3] - b2[3]) * k4 +
                              (b1[4] - b2[4]) * k5 + (b1[5] - b2[5]) * k6 +
                              (b1[6] - b2[6]) * k7))

            h *= 0.9 * np.power(max_error / truncation_error, 1.0 / 5.0)

            if truncation_error <= max_error:
                keep_going = False
        
        time_elapsed += h
        x = new_x
    
    return x