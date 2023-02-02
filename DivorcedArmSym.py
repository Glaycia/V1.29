from sympy import *
init_printing(use_unicode=True)

def translation(theta, xtrans, ytrans):
    return Matrix([[cos(theta), -sin(theta), xtrans], [sin(theta), cos(theta), ytrans], [0, 0, 1]])

time = Symbol("t")
length1 = Symbol("L1")
length2 = Symbol("L2")
mass1 = Symbol("m1")
mass2 = Symbol("m2")
MoI1 = Symbol("I1")
MoI2 = Symbol("I2")
theta1 = Function("theta1")(time)
theta2 = Function("theta2")(time)
symtheta1 = Symbol("\\theta_1")
symtheta2 = Symbol("\\theta_2")
omega1 = Symbol("\\omega_1")
omega2 = Symbol("\\omega_2")
alpha1 = Symbol("\\alpha_1")
alpha2 = Symbol("\\alpha_2")

gravity = Symbol("g")

def convertThetaDerivatives(expression):
    alpha1sub = Derivative(Derivative(theta1))
    alpha2sub = Derivative(Derivative(theta2))
    omega1sub = Derivative(theta1)
    omega2sub = Derivative(theta2)
    return expression.subs(alpha1sub, alpha1).subs(alpha2sub, alpha2).subs(omega1sub, omega1).subs(omega2sub, omega2).subs(theta1, symtheta1).subs(theta2, symtheta2)

T1 = translation(theta1, cos(theta1) * length1, sin(theta1) * length1)
T2 = translation(theta2, cos(theta2) * length2, sin(theta2) * length2)
Transform1 = T1
Transform1Cut = Matrix([Transform1[0, 2], Transform1[1, 2]]) 
Transform2 = T1 + T2
Transform2Cut = Matrix([Transform2[0, 2], Transform2[1, 2]]) 

x1 = Transform1Cut[0, 0]
y1 = Transform1Cut[1, 0]
x2 = Transform2Cut[0, 0]
y2 = Transform2Cut[1, 0]

xdot1 = x1.diff(time)
ydot1 = y1.diff(time)
xdot2 = x2.diff(time)
ydot2 = y2.diff(time)


KE_1 = 0.5 * mass1 * (xdot1 ** 2 + ydot1 ** 2) + 0.5 * (MoI1) * (theta1.diff(time)) ** 2
KE_2 = 0.5 * mass2 * (xdot2 ** 2 + ydot2 ** 2) + 0.5 * (MoI2) * (theta2.diff(time)) ** 2

#KE = ((mass1+mass2) * xdot1 ** 2 + (mass1+mass2) * ydot1 **2 + mass2 * xdot2 ** 2 + mass2 * ydot2 ** 2)/2
KE = KE_1+KE_2
PE = (mass1 * (y1) + mass2 * (y1 + (y2-y1))) * gravity #Change proportion based on CG?

print_latex(convertThetaDerivatives((KE).simplify()))
#print(convertThetaDerivatives(KE.simplify()))
Lagrangian = KE - PE

dTComponent = Matrix([[Lagrangian.diff(theta1.diff(time))], [Lagrangian.diff(theta2.diff(time))]])
dTComponent[0, 0] = dTComponent[0, 0].diff(time)
dTComponent[1, 0] = dTComponent[1, 0].diff(time)

dXComponent = Matrix([[Lagrangian.diff(theta1)], [Lagrangian.diff(theta2)]])

torques = dTComponent - dXComponent
torques[0, 0] = convertThetaDerivatives(torques[0, 0]).simplify()
torques[1, 0] = convertThetaDerivatives(torques[1, 0]).simplify()
print_latex(torques)
