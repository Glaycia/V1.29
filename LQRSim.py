import numpy as np
import control
import time
from numpy import sin, cos

#Arm Parameters
mass_1 = 0.55 #kg
mass_2 = 0.55 #kg
length_1 = 1  #m
length_2 = 0.75  #m
com_1 = 0.5   #m
com_2 = 0.5   #m
inertia_1 = 1 #kg m^2
inertia_2 = 1 #kg m^2
g = 9.806 #m/s^2

def ArmDynamics(x, u):
    # q = [θ1, θ2]ᵀ rad
    # q̇ = [θ̇1, θ̇2]ᵀ rad/s
    # u = [F_1, F_2] N m
    # 
    # M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + u
    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + u)
    
    q = x[0:2]
    qdot = x[2:4]

    # F(thetas) = B(q)@u + C(qdot, q) + g(q)
    M = np.empty((2, 2)) #Mass matrix
    M[0, 0] = inertia_1 + length_1 ** 2 * (mass_1 + 2 * mass_2)
    M[0, 1] = length_1 * length_2 * mass_2 * cos(q[0, 0] - q[1, 0])
    M[1, 0] = length_1 * length_2 * mass_2 * cos(q[0, 0] - q[1, 0])
    M[1, 1] = inertia_2 + length_2 ** 2 * mass_2

    C = np.empty((2, 1)) #Joint Velocity Product Term (Has qdot0**2/qdot1**2 Centripetal terms, but No qdot0*qdot1 Coriolis terms)
    C[0, 0] = length_1 * length_2 * mass_2 * qdot[1, 0]**2 * sin(q[0, 0] - q[1, 0])
    C[1, 0] = -length_1 * length_2 * mass_2 * qdot[0, 0]**2 * sin(q[0, 0] - q[1, 0])

    G = np.empty((2, 1)) #Gravity Term
    G[0, 0] = g * (com_1 * mass_1 * cos(q[0, 0]) + length_2 * mass_2 * cos(q[1, 0]))
    G[1, 0] = g * (com_2 * mass_2 * cos(q[1, 0]))

    #Invert Mass Matrix
    Minv = np.empty((2, 2))
    Minv[0, 0] = M[1, 1]
    Minv[0, 1] = -M[0, 1]
    Minv[1, 0] = -M[1, 0]
    Minv[1, 1] = M[0, 0]
    detM = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    Minv /= detM

    #Return changes in state:
    qddot = np.empty((4, 1))
    qddot[0:2] = qdot
    qddot[2:4] = Minv @ (u - C - G)
    return qddot

def linearizeSystem(x, u):
    epsilon = 1/(2.0 ** 10)
    states = 4
    inputs = 2
    A = np.empty((states, states))
    B = np.empty((states, inputs))
    for i in range(states):
        right = x.copy()
        left = x.copy()
        right[i] = right[i] + epsilon
        left[i] = left[i] - epsilon
        diff = ((ArmDynamics(right, u) - ArmDynamics(left, u))/(2 * epsilon))[:, 0]
        for j in range(states):
            A[j, i] = diff[j]
    for i in range(inputs):
        right = u.copy()
        left = u.copy()
        right[i] += epsilon
        left[i] -= epsilon
        diff = ((ArmDynamics(x, right) - ArmDynamics(x, left))/(2 * epsilon))[:, 0]
        for j in range(states):
            B[j, i] = diff[j]
    #print(A)
    #print(B)
    return A, B

def plantInversion(nextR, r, dT):
    rDot = (nextR-r)/dT
    A, B = linearizeSystem(r, np.zeros((2, 1)))
    #rDot = f(x) + B @ u
    f_x = ArmDynamics(r, np.zeros((2, 1)))
    #print(rDot, ",", f_x)
    return np.linalg.pinv(B) @ (rDot - f_x)

if __name__ == "__main__":
    r = np.empty((4, 1))
    r[0, 0] = 1
    r[1, 0] = 0
    r[2, 0] = 0
    r[3, 0] = 0

    x = np.empty((4, 1))
    x[0, 0] = 1.2
    x[1, 0] = -1
    x[2, 0] = 0.1
    x[3, 0] = 0.05

    u = np.empty((2, 1))
    u[0, 0] = 0
    u[1, 0] = 0

    deviationTheta = 0.01
    deviationOmega = 0.1
    Q = np.zeros((4, 4))
    Q[0, 0] = 1/deviationTheta**2
    Q[1, 1] = 1/deviationTheta**2
    Q[2, 2] = 1/deviationOmega**2
    Q[3, 3] = 1/deviationOmega**2

    controlCost = 10
    R = np.zeros((2, 2))
    R[0, 0] = controlCost
    R[1, 1] = controlCost

    A, B = linearizeSystem(x, u)

    K, S, E = control.lqr(A, B, Q, R)

    #print(K)
    app_u = K @ (r - x) + plantInversion(r, r)

    #print(app_u)