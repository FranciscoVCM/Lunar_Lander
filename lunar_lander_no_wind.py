import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = False
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

def too_far_left(obs):
    return obs[0] < -0.3

def too_far_right(obs):
    return obs[0] > 0.3

def slightly_left(obs):
    return -0.3 <= obs[0] < -0.1

def slightly_right(obs):
    return 0.1 < obs[0] <= 0.3

def centered(obs):
    return abs(obs[0]) <= 0.1

def moving_left_fast(obs):
    return obs[2] < -0.4

def moving_right_fast(obs):
    return obs[2] > 0.4

def moving_left(obs):
    return obs[2] < -0.15

def moving_right(obs):
    return obs[2] > 0.15

def falling_critical(obs):
    return obs[3] < -0.8

def falling_fast(obs):
    return -0.8 <= obs[3] < -0.4

def falling_moderate(obs):
    return -0.4 <= obs[3] < -0.2

def falling_slow(obs):
    return obs[3] >= -0.2

def tilted_left_strong(obs):
    return obs[4] > 0.35

def tilted_right_strong(obs):
    return obs[4] < -0.35

def tilted_left(obs):
    return 0.15 < obs[4] <= 0.35

def tilted_right(obs):
    return -0.35 <= obs[4] < -0.15

def upright(obs):
    return abs(obs[4]) <= 0.15

def rotating_left_fast(obs):
    return obs[5] > 0.3

def rotating_right_fast(obs):
    return obs[5] < -0.3

def very_high(obs):
    return obs[1] > 1.2

def high(obs):
    return 0.6 < obs[1] <= 1.2

def medium_height(obs):
    return 0.3 < obs[1] <= 0.6

def low(obs):
    return obs[1] <= 0.3

def both_legs_touching(obs):
    return obs[6] == 1 and obs[7] == 1

def lateral_error(obs, anticipation=1.8):
    return obs[0] + obs[2] * anticipation

def needs_to_go_right(obs, threshold=0.12):
    return lateral_error(obs) < -threshold

def needs_to_go_left(obs, threshold=0.12):
    return lateral_error(obs) > threshold

def lateral_error_large(obs):
    return abs(lateral_error(obs)) > 0.45

def lateral_error_medium(obs):
    return 0.2 < abs(lateral_error(obs)) <= 0.45

def lateral_aligned(obs):
    return abs(lateral_error(obs)) <= 0.12


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


# -------------------------
# REACTIVE AGENT
# -------------------------

def reactive_agent(observation):
    x   = observation[0]
    y   = observation[1]
    vx  = observation[2]
    vy  = observation[3]
    th  = observation[4]
    vth = observation[5]

    # CASO EMERGÊNCIA
    if vy < -1.2 and abs(th) < 0.5:
        return full_thrust()
    if vy < -0.9 and abs(th) < 0.4:
        return strong_thrust()

    # ESTABILIZAR ROTAÇÃO EXCESSIVA
    if vth > 0.35:
        return rotate_right_strong()
    if vth < -0.35:
        return rotate_left_strong()

    # REGRA 3 — INCLINAÇÃO SEVERA
    if th > 0.35:
        t = 0.5 if vy < -0.4 else 0.0
        return np.array([t, 0.8])
    if th < -0.35:
        t = 0.5 if vy < -0.4 else 0.0
        return np.array([t, -0.8])

    # CONTROLO DE DESCIDA + HOVER LATERAL
    if y > 1.2:
        vy_limit = -0.7
    elif y > 0.6:
        vy_limit = -0.45
    elif y > 0.3:
        vy_limit = -0.25
    else:
        vy_limit = -0.12

    if y > 1.0:
        k = 3.5
    elif y > 0.5:
        k = 2.8
    else:
        k = 1.2
    err = x + vx * k

    if y > 0.5 and abs(err) > 0.4:
        hover_factor = np.clip((abs(err) - 0.4) / 0.4, 0.0, 1.0)
        vy_limit = vy_limit * (1.0 - hover_factor) + 0.05 * hover_factor

    if y > 0.5:
        if y > 0.8:
            target_th_r4 = np.clip(err * 0.38, -0.30, 0.30)
        else:
            target_th_r4 = np.clip(err * 0.30, -0.26, 0.26)
    else:
        fade = y / 0.5
        target_th_r4 = np.clip(err * 0.18 * fade, -0.10 * fade, 0.10 * fade)
        if 0.15 < abs(x) <= 0.40 and y > 0.15:
            precision_th = np.sign(x) * 0.06
            if abs(target_th_r4) < abs(precision_th):
                target_th_r4 = precision_th
        if abs(x) < 0.25 and abs(th) < 0.12:
            target_th_r4 += np.clip(-vth * 0.4, -0.05, 0.05)

    th_err_r4 = th - target_th_r4
    r4_dead = 0.03 if y < 0.5 else 0.05
    r4_strong = 0.08 if y < 0.5 else 0.15
    if th_err_r4 > r4_dead and vth > -0.18:
        rot_r4 = 0.6 if (th_err_r4 > r4_strong or vth > 0.12) else 0.55
    elif th_err_r4 < -r4_dead and vth < 0.18:
        rot_r4 = -0.6 if (th_err_r4 < -r4_strong or vth < -0.12) else -0.55
    else:
        rot_r4 = 0.0

    if vy < vy_limit:
        excess = vy_limit - vy
        if excess > 0.6:
            thrust = 1.0
        elif excess > 0.35:
            thrust = 0.8
        elif excess > 0.15:
            thrust = 0.55
        else:
            thrust = 0.35
        return np.array([thrust, rot_r4])

    # CONTROLO LATERAL + ENDIREITAR
    if y > 1.0:
        k = 3.5
    elif y > 0.5:
        k = 2.8
    else:
        k = 1.2
    err = x + vx * k

    if y > 0.5:
        if y > 0.8:
            target_th = np.clip(err * 0.38, -0.30, 0.30)
        else:
            target_th = np.clip(err * 0.30, -0.26, 0.26)
    else:
        fade = y / 0.5
        target_th = np.clip(err * 0.18 * fade, -0.10 * fade, 0.10 * fade)
        if 0.15 < abs(x) <= 0.40 and y > 0.15:
            precision_th = np.sign(x) * 0.06
            if abs(target_th) < abs(precision_th):
                target_th = precision_th
        if abs(x) < 0.25 and abs(th) < 0.12:
            target_th += np.clip(-vth * 0.4, -0.05, 0.05)

    th_err = th - target_th

    if y < 0.5:
        dead_zone = 0.03
        strong_threshold = 0.08
    else:
        dead_zone = 0.05
        strong_threshold = 0.15

    if th_err > dead_zone and vth > -0.18:
        if th_err > strong_threshold or vth > 0.12:
            return rotate_right_soft()
        elif y < 0.30:
            return np.array([0.0, 0.51])
        else:
            return rotate_right_gentle()

    if th_err < -dead_zone and vth < 0.18:
        if th_err < -strong_threshold or vth < -0.12:
            return rotate_left_soft()
        elif y < 0.30:
            return np.array([0.0, -0.51])
        else:
            return rotate_left_gentle()

    return do_nothing()


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
