import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = True
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human'
RENDER_MODE = None
EPISODES = 1000

env = gym.make(
    "LunarLander-v3",
    render_mode=RENDER_MODE,
    continuous=True,
    gravity=GRAVITY,
    enable_wind=ENABLE_WIND,
    wind_power=WIND_POWER,
    turbulence_power=TURBULENCE_POWER
)

def check_successful_landing(observation):
    x          = observation[0]
    vy         = observation[3]
    theta      = observation[4]
    contact_left  = observation[6]
    contact_right = observation[7]

    legs_touching      = contact_left == 1 and contact_right == 1
    on_landing_pad     = abs(x) <= 0.2
    stable_velocity    = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)

    return legs_touching and on_landing_pad and stable_velocity and stable_orientation


def simulate(steps=1000, seed=None, policy=None):
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)
        observ, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    return step, check_successful_landing(observ)


# -------------------------
# PERCEPTIONS
# -------------------------

def x_pos(obs):
    return obs[0]


def y_pos(obs):
    return obs[1]


def x_vel(obs):
    return obs[2]


def y_vel(obs):
    return obs[3]


def tilt(obs):
    return obs[4]


def ang_vel(obs):
    return obs[5]


def left_leg_touching(obs):
    return obs[6] == 1


def right_leg_touching(obs):
    return obs[7] == 1


def both_legs_touching(obs):
    return left_leg_touching(obs) and right_leg_touching(obs)


def one_leg_touching(obs):
    return left_leg_touching(obs) != right_leg_touching(obs)


def altitude_band(obs):
    y = y_pos(obs)
    if y > 1.0:
        return "very_high"
    if y > 0.6:
        return "high"
    if y > 0.3:
        return "medium"
    return "low"


def lateral_prediction(obs):
    y = y_pos(obs)
    if y > 1.0:
        horizon = 3.8
    elif y > 0.6:
        horizon = 3.0
    elif y > 0.3:
        horizon = 2.0
    else:
        horizon = 1.7

    x = x_pos(obs)
    vx = x_vel(obs)
    x_future = x + vx * horizon

    if abs(x) > 0.22 and np.sign(x) == np.sign(vx):
        x_future += np.sign(x) * 0.16 * min(abs(vx), 1.0)

    return x_future


def lateral_pressure(obs):
    return abs(lateral_prediction(obs)) + 0.95 * abs(x_vel(obs))


def emergency_fall(obs):
    return y_vel(obs) < -1.15 and abs(tilt(obs)) < 0.55


def heavy_fall(obs):
    return y_vel(obs) < -0.90 and abs(tilt(obs)) < 0.45


# -------------------------
# ACTIONS
# -------------------------

def do_nothing():
    return np.array([0.0, 0.0])

def full_thrust():
    return np.array([1.0, 0.0])

def strong_thrust():
    return np.array([0.8, 0.0])

def medium_thrust():
    return np.array([0.55, 0.0])

def light_thrust():
    return np.array([0.35, 0.0])

def hover_thrust():
    return np.array([0.2, 0.0])

def rotate_right_strong():
    return np.array([0.0, 0.8])

def rotate_left_strong():
    return np.array([0.0, -0.8])

def rotate_right_soft():
    return np.array([0.0, 0.6])

def rotate_left_soft():
    return np.array([0.0, -0.6])

def rotate_right_gentle():
    return np.array([0.0, 0.55])

def rotate_left_gentle():
    return np.array([0.0, -0.55])

def thrust_and_rotate_right(t=0.4, r=0.6):
    return np.array([t, r])

def thrust_and_rotate_left(t=0.4, r=-0.6):
    return np.array([t, r])


def thrust_and_rotate(thrust, rotation):
    return np.array([float(np.clip(thrust, 0.0, 1.0)), float(np.clip(rotation, -1.0, 1.0))])


# -------------------------
# REACTIVE AGENT
# -------------------------

def reactive_agent(observation):
    if both_legs_touching(observation):
        return do_nothing()

    ##Travao de queda
    if emergency_fall(observation):
        return full_thrust()
    if heavy_fall(observation):
        return strong_thrust()

    x = x_pos(observation)
    y = y_pos(observation)
    vx = x_vel(observation)
    vy = y_vel(observation)
    th = tilt(observation)
    vth = ang_vel(observation)

    ## Escolher o theta com base na previsao dp erro lateral
    x_future = lateral_prediction(observation)
    band = altitude_band(observation)

    if band == "very_high":
        target_th = np.clip(0.42 * x_future, -0.38, 0.38)
    elif band == "high":
        target_th = np.clip(0.32 * x_future, -0.26, 0.26)
    elif band == "medium":
        target_th = np.clip(0.24 * x_future, -0.16, 0.16)
    else:
        fade = np.clip(y / 0.30, 0.45, 1.0)
        target_th = np.clip(0.22 * x_future * fade, -0.14 * fade, 0.14 * fade)

    
    if y < 0.30 and (abs(x) > 0.11 or abs(vx) > 0.11):
        target_th = np.clip(0.55 * x + 1.20 * vx, -0.22, 0.22)

    one_leg = one_leg_touching(observation)
    if one_leg_touching(observation):
        target_th = 0.0

    # Controlo angular     
    th_err = th - target_th
    rot_raw = 2.05 * th_err + 0.78 * vth
    if abs(rot_raw) < 0.04:
        rot_cmd = 0.0
    else:
        rot_cmd = np.sign(rot_raw) * np.clip(0.51 + 0.43 * abs(rot_raw), 0.51, 1.0)

    if y < 0.20:
        rot_cmd *= 1.05

    ## Objetivo vertical adaptativo
    if y > 1.2:
        vy_target = -0.62
    elif y > 0.75:
        vy_target = -0.46
    elif y > 0.40:
        vy_target = -0.30
    elif y > 0.20:
        vy_target = -0.20
    else:
        vy_target = -0.10

    pressure = lateral_pressure(observation)
    if y > 0.35:
        if pressure > 0.58:
            vy_target = max(vy_target, -0.02)
        elif pressure > 0.36:
            vy_target = max(vy_target, -0.08)

    if one_leg:
        vy_target = max(vy_target, -0.03)

    if y < 0.35:
        if abs(x) > 0.12 or abs(vx) > 0.12:
            vy_target = max(vy_target, 0.00)
        elif abs(x) > 0.07 or abs(vx) > 0.08:
            vy_target = max(vy_target, -0.03)

    if y < 0.22 and (abs(x) > 0.09 or abs(vx) > 0.10):
        vy_target = max(vy_target, 0.02)

    ## Conversão erro vertical 
    excess = vy_target - vy
    if excess > 0.75:
        thrust = 1.0
    elif excess > 0.48:
        thrust = 0.85
    elif excess > 0.26:
        thrust = 0.60
    elif excess > 0.10:
        thrust = 0.38
    elif excess > 0.04:
        thrust = 0.24
    else:
        thrust = 0.0

    if y < 0.18 and vy < -0.10:
        thrust = max(thrust, 0.32)

    return thrust_and_rotate(thrust, rot_cmd)


success = 0.0
steps = 0.0

for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)
    if su:
        steps += st
    success += su
    if su > 0:
        print("Média de passos das aterragens bem sucedidas:", steps / success)
    print("Taxa de sucesso:", success / (i + 1) * 100)
