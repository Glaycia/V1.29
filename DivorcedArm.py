import casadi
import numpy as np
from casadi import MX, sin, cos, fabs, exp

def RungeKutta4(dyn, x, u, dT):
    k1 = dyn(x, u)
    k2 = dyn(x + dT * 0.5 * k1, u)
    k3 = dyn(x + dT * 0.5 * k2, u)
    k4 = dyn(x + dT * k3, u)

    return x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def Integration(dyn, x, u, dT):
    return x + dyn(x, u)*dT

mass_1 = 0.55 #kg
mass_2 = 0.55 #kg
length_1 = 1  #m
length_2 = 0.75  #m
com_1 = 0.5   #m
com_2 = 0.5   #m
inertia_1 = 1 #kg m^2
inertia_2 = 1 #kg m^2
g = 9.806 #m/s^2

def ArmDynamics(x = MX, u = MX):
    # q = [θ1, θ2]ᵀ rad
    # q̇ = [θ̇1, θ̇2]ᵀ rad/s
    # u = [F_1, F_2] N m
    # 
    # M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + u
    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + u)
    
    q = x[0:2, 0]
    qdot = x[2:4, 0]

    # F(thetas) = B(q)@u + C(qdot, q) + g(q)
    M = MX(2, 2) #Mass matrix
    M[0, 0] = inertia_1 + length_1 ** 2 * (mass_1 + 2 * mass_2)
    M[0, 1] = length_1 * length_2 * mass_2 * cos(q[0, 0] - q[1, 0])
    M[1, 0] = length_1 * length_2 * mass_2 * cos(q[0, 0] - q[1, 0])
    M[1, 1] = inertia_2 + length_2 ** 2 * mass_2

    C = MX(2, 1) #Joint Velocity Product Term (Has qdot0**2/qdot1**2 Centripetal terms, but No qdot0*qdot1 Coriolis terms)
    C[0, 0] = length_1 * length_2 * mass_2 * qdot[1, 0]**2 * sin(q[0, 0] - q[1, 0])
    C[1, 0] = -length_1 * length_2 * mass_2 * qdot[0, 0]**2 * sin(q[0, 0] - q[1, 0])

    G = MX(2, 1) #Gravity Term
    G[0, 0] = g * (com_1 * mass_1 * cos(q[0, 0]) + length_2 * mass_2 * cos(q[0, 0]))
    G[1, 0] = g * (com_2 * mass_2 * cos(q[1, 0]))

    #Invert Mass Matrix
    Minv = MX(2, 2)
    Minv[0, 0] = M[1, 1]
    Minv[0, 1] = -M[0, 1]
    Minv[1, 0] = -M[1, 0]
    Minv[1, 1] = M[0, 0]
    detM = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    Minv /= detM

    #Return changes in state:
    qddot = MX(4, 1)
    qddot[0:2, 0] = qdot
    qddot[2:4, 0] = Minv @ (u)
    return qddot

def ForwardKinematics(x = MX):
    q = x[0:2, 0]
    xy = MX(2, 1)
    xy[0, 0] = length_1 * cos(q[0, 0]) + length_2 * cos(q[1, 0])
    xy[1, 0] = length_1 * sin(q[0, 0]) + length_2 * sin(q[1, 0])
    return xy
def CartesianBoundsViolation(x = MX):
    xy = ForwardKinematics(x)
    steepness = 150
    boundaryX = 1.0 #not inbetween +- this value
    boundaryY = 0.5 #above this value

    xcost = 1/((1+exp(-steepness * (xy[0, 0] + boundaryX))) * (1+exp(steepness * (xy[0, 0] - boundaryX))))
    ycost = 1/(1+exp(steepness * (xy[1, 0] - boundaryY)))
    return xcost + ycost < 1.1

def DivorcedArm(N):
    solver = casadi.Opti()
    solver.solver('ipopt')

    PI = casadi.pi
    u_max =    20   #N m
    J1_min =   PI/6 #rad
    J1_max = 5*PI/6 #rad
    J2_min =-3*PI/2 #rad
    J2_max =   PI/2 #rad

    J1_init =  PI/4 #rad
    J2_init = 0 #rad
    J1_target = 3*PI/4  #rad
    J2_target = -PI/2  #rad

    min_dT = 0.025
    max_dT = 0.075
    scaling_step = casadi.power(5, 1/N) #How much each step should multiplicatevely increase, the last step is x times longer.

    X = solver.variable(4, N + 1)
    for k in range(N):
        solver.set_initial(X[0, k], J1_init + k / N * (J1_target - J1_init))
        solver.set_initial(X[1, k], J2_init + k / N * (J2_target - J2_init))
    U = solver.variable(2, N)
    timesteps = solver.variable(1, N)

    #Initial conditions
    solver.subject_to(X[0, 0] == J1_init)
    solver.subject_to(X[1, 0] == J2_init)
    solver.subject_to(X[2:4, 0] == 0)

    #Final conditions
    solver.subject_to(X[0, N] == J1_target)
    solver.subject_to(X[1, N] == J2_target)
    solver.subject_to(X[2:4, N] == 0)

    #Positional Constrains
    solver.subject_to(solver.bounded(J1_min, X[0, :], J1_max))
    solver.subject_to(solver.bounded(J2_min, X[1, :], J2_max))
    #for k in range(N):
    #    solver.subject_to(CartesianBoundsViolation(X[:, k]))

    #Control Constraints
    solver.subject_to(solver.bounded(-u_max, U[0, :], u_max))
    solver.subject_to(solver.bounded(-u_max, U[1, :], u_max))

    for k in range(N):
        lb_dT = min_dT #* casadi.power(scaling_step, N)
        hb_dT = max_dT #* casadi.power(scaling_step, N)
        solver.subject_to(solver.bounded(lb_dT, timesteps[0, k], hb_dT)) #Time

        solver.subject_to(X[:, k+1] == Integration(ArmDynamics, X[:, k], U[:, k], timesteps[0, k])) #Dynamics
    
    J = 0
    #Minimize time
    for k in range(N):
        J += timesteps[0, k]
    
    solver.minimize(J)

    return solver, X, U, timesteps

if __name__ == "__main__":
    N = 10
    solver, X, U, TS = DivorcedArm(N)
    solution = solver.solve()
    state_values = solution.value(X)
    timesteps = solution.value(TS)
    integratedTime = np.zeros(N+1)
    for i in range(N):
        sum = 0
        for j in range(i):
            sum += timesteps[j]
        integratedTime[i+1] = sum
    state_position = state_values[0:2, :]
    state_velocity = state_values[2:4, :]
    for i in range(N+1):
        q = (state_position[:, i:i+1])
        x1 = length_1 * cos(q[0, 0])
        y1 = length_1 * sin(q[0, 0])
        print("(", x1, ", ", y1, ")")
        x2 = length_1 * cos(q[0, 0]) + length_2 * cos(q[1, 0])
        y2 = length_1 * sin(q[0, 0]) + length_2 * sin(q[1, 0])
        print("(", x2, ", ", y2, ")")
        #print("(", integratedTime[i], ", ", q[0, 0], ")")
        #print("(", integratedTime[i], ", ", q[1, 0], ")")

        qdot = (state_velocity[:, i:i+1])
        #print("(", integratedTime[i], ", ", qdot[0, 0], ")")
        #print("(", integratedTime[i], ", ", qdot[1, 0], ")")
        print("()")
        print("()")

        print("()")