import os

import gymnasium as gym
import imageio
import numpy as np


# Take from
# https://github.com/denisyarats/pytorch_sac/
class VideoRecorder(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        save_folder: str = "",
        height: int = 128,
        width: int = 128,
        fps: int = 30,
        camera_id: int = 0,
    ):
        super().__init__(env)

        self.current_episode = 0
        self.save_folder = save_folder
        self.height = height
        self.width = width
        self.fps = fps
        self.camera_id = camera_id
        self.frames = []

        try:
            os.makedirs(save_folder, exist_ok=True)
        except:
            pass

    def step(self, action: np.ndarray):

        frame = self.env.render(
            mode="rgb_array",
            height=self.height,
            width=self.width,
            camera_id=self.camera_id,
        )

        if frame is None:
            try:
                frame = self.sim.render(
                    width=self.width, height=self.height, mode="offscreen"
                )
                frame = np.flipud(frame)
            except:
                raise NotImplementedError("Rendering is not implemented.")

        self.frames.append(frame)

        observation, reward, done, info = self.env.step(action)

        if done:
            save_file = os.path.join(self.save_folder, f"{self.current_episode}.mp4")
            imageio.mimsave(save_file, self.frames, fps=self.fps)
            self.frames = []
            self.current_episode += 1

        return observation, reward, done, info
