import numpy
import pybullet as p
import time
import pybullet_data
import numpy as np
from gymnasium.spaces import Box
from gymnasium import Env, spaces
from gymnasium.utils import env_checker
from stable_baselines3 import SAC, PPO, A2C, DDPG, TD3
import random
import os
import math
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
import pygame

distance = 100000
img_w, img_h = 32, 32
num_of_frames = 5


#2- front left
#3- front right
#4- back left
#5- back right

class hEnv(Env):
    def __init__(self):
        super(hEnv, self).__init__()
        p.connect(p.GUI)
        self.observation_space = Box(low=0, high=255, shape=(num_of_frames-1,3, img_h, img_w), dtype=np.uint8)
        self.action_space = Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.steps = 0
        self.obs = [np.zeros((3, img_h, img_w), dtype=np.uint8) for _ in range(num_of_frames - 1)]
        print(np.array(self.obs).shape)
        self.xbox()
        if pygame.joystick.get_count() == 0:
            print("No joystick")

    def xbox(self):
        pygame.init()
        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

    def load_model(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        self.model = p.loadURDF("husky/husky.urdf", basePosition=[0, 0, 0])
        self.walls = p.loadURDF("D:/Proiecte/HumanoidAI/HumanoidAI/URDF/walls/walls.urdf", basePosition=[0, 0, 0])
        for i in range(p.getNumJoints(self.model)):
            print(p.getJointInfo(self.model,i))
        p.setGravity(0,0,-9.8)

    def step(self, action):
        reward, done = self.get_reward_done()
        obs = self.get_observation()
        for i in range(2, 6):
            p.setJointMotorControl2(self.model, i, p.VELOCITY_CONTROL, targetVelocity=action[i - 2])
            p.stepSimulation()
        p.stepSimulation()
        self.steps += 1
        info = {}
        truncated = False
        print(reward)
        return obs, reward, done, truncated, info

    def controll(self):

        pygame.event.get()
        left_joystick_value = pygame.joystick.Joystick(0).get_axis(1) * -10
        right_joystick_value = pygame.joystick.Joystick(0).get_axis(3) * -10

        print(left_joystick_value, right_joystick_value)

        output = [left_joystick_value, right_joystick_value, left_joystick_value, right_joystick_value]
        return output

    def make_photo(self):
        agent_pos, agent_orn = list(p.getLinkState(self.model, 8, computeForwardKinematics=True))[:2]
        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        xA, yA, zA = agent_pos
        zA = zA + 0.3
        xB = xA + math.cos(yaw) * distance
        yB = yA + math.sin(yaw) * distance
        zB = zA
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[xA, yA, zA],
            cameraTargetPosition=[xB, yB, zB],
            cameraUpVector=[0, 0, 1.0]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=75, aspect=1.5, nearVal=0.02, farVal=25)
        self.rgb_image = p.getCameraImage(img_w, img_h, view_matrix, projection_matrix, shadow=True,
                                          renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # Compute the projection matrix with the new camera parameters
        # self.rgb_image = p.getCameraImage(img_w, img_h,shadow=True,
        #                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.rgb_imageF = np.reshape(self.rgb_image[2], [4, img_h, img_w])
        self.rgb_imageF = self.rgb_imageF[:3, :, :]
        return self.rgb_imageF

    def get_observation(self):
        self.obs.pop(0)
        self.obs.append(self.make_photo())
        return self.obs

    def spawn_target(self):
        random_x = random.uniform(-5.0,5.0)
        random_y = random.uniform(-5.0, 5.0)
        if random_x < 2 and random_x>0:random_x=2
        if random_x >-2 and random_x<0:random_x=-2
        if random_y < 2 and random_y>0:random_y=2
        if random_y >-2 and random_y<0:random_y=-2

        # Set the third component to 0
        base_position = [random_x, random_y, 0.5]

        self.target = p.loadURDF('cubeRed.urdf',basePosition=base_position)

    def num_to_range(self,num, inMin, inMax, outMin, outMax):
        return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))
    def get_reward_done(self):
        joint_info = p.getBasePositionAndOrientation(self.model)
        x1,y1,_ = joint_info[0]

        base_position = p.getBasePositionAndOrientation(self.target)
        x2,y2,_ = base_position[0]

        distance = abs(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        done = False

        reward = distance

        if distance < 1.25:
            done = True
            reward = 10-distance
        # if p.getContactPoints(self.model, self.walls):
        #     done = True
        #     reward = -15
        if self.steps >= 1000:
            done = True
            reward = 5-distance
        return reward, done

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if p.isConnected():
            p.disconnect()
            time.sleep(1)
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=7.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])
        self.steps = 0
        self.load_model()
        self.spawn_target()
        self.xbox()
        obs = [self.make_photo() for _ in range(num_of_frames-1)]
        info = {}
        return obs, info

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


env = hEnv()

env_v = DummyVecEnv([lambda: env])

CHECKPOINT_DIR = './train2/'
LOG_DIR = './logs2/'


callback = TrainAndLoggingCallback(check_freq=25000, save_path=CHECKPOINT_DIR)

model = SAC("MlpPolicy", env_v, verbose=1, tensorboard_log=LOG_DIR, batch_size=64)

# model = model.load('best_model_5000000')
#
# model.set_env(env)

model.learn(total_timesteps=5000000, callback=callback)


