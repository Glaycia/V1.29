# Import and initialize the pygame library
import pygame
import numpy as np

import Kinematics
import LQRSim

pygame.init()

clock = pygame.time.Clock()

pixelsPerMeter = 160
screenDim = (4 * pixelsPerMeter, 4 * pixelsPerMeter)
# Set up the drawing window
screen = pygame.display.set_mode(screenDim)

previousMouseState = (0, 0, 0)
class SimState:
    targetT1 = 0
    targetT2 = 0
    targetX = LQRSim.length_1 + LQRSim.length_2
    targetY = 0
    currentT1 = 0
    currentT2 = 0
    currentW1 = 0
    currentW2 = 0

    previousR = None

state = SimState()

def registerMouse(previousMouseState, arm):
    mX, mY = pygame.mouse.get_pos()
    centerCoords = (screenDim[0]/2, screenDim[1]/2)
    metersX = (mX - centerCoords[0])/pixelsPerMeter
    metersY = (centerCoords[1] - mY)/pixelsPerMeter

    mouseDist = np.hypot(metersX, metersY)
    if (mouseDist > LQRSim.length_1 + LQRSim.length_2):
        metersX *= (LQRSim.length_1 + LQRSim.length_2)/mouseDist
        metersY *= (LQRSim.length_1 + LQRSim.length_2)/mouseDist
    if (mouseDist < np.abs(LQRSim.length_1 - LQRSim.length_2)):
        metersX *= np.abs(LQRSim.length_1 - LQRSim.length_2)/mouseDist
        metersY *= np.abs(LQRSim.length_1 - LQRSim.length_2)/mouseDist

    if pygame.mouse.get_pressed() == (1, 0, 0) and previousMouseState == (0, 0, 0):
        angles = Kinematics.inverseKinematics(metersX, metersY, LQRSim.length_1, LQRSim.length_2)
        arm.targetX = metersX
        arm.targetY = metersY
        arm.targetT1 = angles[0]
        arm.targetT2 = angles[1]
    
    previousMouseState = pygame.mouse.get_pressed()

def drawArm(theta1, theta2, length1, length2, linewidth, color):
    centerCoords = (screenDim[0]/2, screenDim[1]/2)
    coordsJ1 = (centerCoords[0] + np.cos(theta1) * length1 * pixelsPerMeter, centerCoords[1] - np.sin(theta1) * length1 * pixelsPerMeter)
    coordsJ2 = (coordsJ1[0] + np.cos(theta2) * length2 * pixelsPerMeter, coordsJ1[1] - np.sin(theta2) * length2 * pixelsPerMeter)
    #pygame.draw.rect(screen, (0, 255, 0), pygame.Rect((j * cellWidth, i * cellHeight), (cellWidth, cellHeight)))
    pygame.draw.line(screen, color, centerCoords, coordsJ1, width=linewidth)
    pygame.draw.line(screen, color, coordsJ1, coordsJ2, width=linewidth)

    pygame.draw.circle(screen, color, coordsJ1, linewidth*2)
    pygame.draw.circle(screen, color, coordsJ2, linewidth*2)
    pass

def bind360(x):
    #set within +pi -pi
    while x > np.pi:
        x -= 2 * np.pi
    while x < -np.pi:
        x += 2 * np.pi
    return x
def iterateState(arm: SimState, u, dT):
    x = np.empty((4, 1))
    x[0] = arm.currentT1
    x[1] = arm.currentT2
    x[2] = arm.currentW1
    x[3] = arm.currentW2

    dynamics = LQRSim.ArmDynamics(x, u)[:, 0]
    #print("Dyn", dynamics)
    arm.currentT1 += dynamics[0] * dT
    arm.currentT2 += dynamics[1] * dT
    arm.currentW1 += dynamics[2] * dT
    arm.currentW2 += dynamics[3] * dT

    #kFriction = 0.1
    #arm.currentW1 -= np.sign(arm.currentW1) * kFriction * dT
    #arm.currentW2 -= np.sign(arm.currentW2) * kFriction * dT

    #print("Arm T1", arm.currentT1)

# Run until the user asks to quit
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (0, 255, 255), (state.targetX * pixelsPerMeter + screenDim[0]/2, -state.targetY * pixelsPerMeter + screenDim[1]/2), 13)
    registerMouse(previousMouseState, state)
    


    r = np.empty((4, 1))
    r[0, 0] = state.targetT1
    r[1, 0] = state.targetT2
    r[2, 0] = 0
    r[3, 0] = 0
    if(state.previousR is None):
        state.previousR = r.copy()
    
    x = np.empty((4, 1))
    x[0, 0] = state.currentT1
    x[1, 0] = state.currentT2
    x[2, 0] = state.currentW1
    x[3, 0] = state.currentW2

    tpX = LQRSim.length_1 * np.cos(r[0, 0]) + LQRSim.length_2 * np.cos(r[1, 0])
    cpX = LQRSim.length_1 * np.cos(x[0, 0]) + LQRSim.length_2 * np.cos(x[1, 0])
    targetIsLeft = tpX > 0
    stateIsLeft = cpX > 0

    if(targetIsLeft != stateIsLeft):
        r[0, 0] = np.pi/2
        r[1, 0] = -np.pi/2
        if(targetIsLeft):
            r[2, 0] = -1
            r[3, 0] = 5
        else:
            r[2, 0] = 1
            r[3, 0] = -5

    u = np.empty((2, 1))
    u[0, 0] = 0
    u[1, 0] = 0

    deviationTheta = 0.001
    deviationOmega = 0.01
    Q = np.zeros((4, 4))
    Q[0, 0] = 1/deviationTheta**2
    Q[1, 1] = 1/deviationTheta**2
    Q[2, 2] = 1/deviationOmega**2
    Q[3, 3] = 1/deviationOmega**2

    controlCost = 10
    R = np.zeros((2, 2))
    R[0, 0] = controlCost
    R[1, 1] = controlCost

    A, B = LQRSim.linearizeSystem(x, u)

    K, S, E = LQRSim.control.lqr(A, B, Q, R)

    error = r - x
    error[0, 0] = bind360(error[0, 0])
    error[1, 0] = bind360(error[1, 0])
    #print(error)

    dT = clock.tick(60)/1000
    app_u = K @ (error) + LQRSim.plantInversion(r, state.previousR, dT)
    #print(LQRSim.plantInversion(r, r, dT))

    iterateState(state, app_u, dT)
    state.previousR = r.copy()
    drawArm(state.targetT1, state.targetT2, LQRSim.length_1, LQRSim.length_2, 6, (0, 200, 200))
    drawArm(state.currentT1, state.currentT2, LQRSim.length_1, LQRSim.length_2, 5, (50, 50, 50))

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()