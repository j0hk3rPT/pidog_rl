import mujoco
import mujoco.viewer
import time
from pidog_env import PiDogEnv

env = PiDogEnv()
model = env.model
data = env.data

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer open — press ESC to exit")
    start = time.time()
    sim_start = data.time

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

        # Keep simulation close to real time
        time_until_next_step = (data.time - sim_start) - (time.time() - start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
