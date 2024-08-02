import numpy as np
from collections import deque
from gymnasium.envs.registration import register
import gymnasium as gym
import cv2


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros(shape=(2, *self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops

    def step(self, action):
        t_reward = 0.0
        done = False
        truncated = False

        for i in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)

            if self.clip_reward:
                reward = np.clip(np.array([reward]), -90, 90)[0]

            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done or truncated:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])

        return max_frame, t_reward, done, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _, _ = self.env.step(0)
            if done:
                obs, info = self.env.reset()

        self.frame_buffer = np.zeros(shape=(2, *self.shape))
        self.frame_buffer[0] = obs

        return obs, info


class PreprocessFrame(gym.ObservationWrapper):

    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)

        self.shape = (shape[2], shape[0], shape[1])

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.shape, dtype=np.float32
        )

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(
            new_frame, self.shape[1:], interpolation=cv2.INTER_AREA
        )

        new_obs = np.array(resized_screen, dtype=np.float32).reshape(self.shape)
        new_obs /= 255.0

        return new_obs


class StackFrames(gym.ObservationWrapper):

    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)

        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32,
        )

        self.stack = deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation, info = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        arrays = [item[0] for item in self.stack]
        return np.array(arrays).reshape(self.observation_space.low.shape), info

    def observation(self, observation):
        self.stack.append(observation)
        arrays = [item[0] for item in self.stack]

        return np.array(arrays).reshape(self.observation_space.low.shape)


def register_env(env_id, entrypoint):
    return register(id=env_id, entry_point=entrypoint)


def make_env(
    env_name,
    entrypoint,
    render_mode="human",
    shape=(84, 84, 1),
    repeat=4,
    clip_rewards=True,
    no_ops=0,
):
    register_env(env_name, entrypoint)
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    return env
