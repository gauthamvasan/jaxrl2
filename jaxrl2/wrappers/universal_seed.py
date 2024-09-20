import gym


class UniversalSeed(gym.Wrapper):
    def seed(self, seed: int):
        try:
            seeds = self.env.seed(seed)
        except Exception as e:
            seeds = None
        self.env.observation_space.seed(seed)
        self.env.action_space.seed(seed)
        return seeds
