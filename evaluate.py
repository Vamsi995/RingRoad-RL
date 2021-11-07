from matplotlib import pyplot as plt

from metrics import Metrics
from simulator import Env

def main():
    TIME_STEPS = 500
    env = Env()
    env.reset(21, 1)
    met = Metrics(env)
    t = 0
    met.register_cars()
    while t < TIME_STEPS:
        env.step()
        met.store_xy(t)
        met.store_v(t)
        met.running_mean_vel()
        t += 1

    met.plot_velocities()
    # met.plot_positions()

if __name__ == "__main__":
    main()