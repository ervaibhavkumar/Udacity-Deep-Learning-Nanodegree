import numpy as np
import math
from physics_sim import PhysicsSim

class new_task():
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):

        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4


        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.]) 

    def get_reward(self):

        penalty = (1. - abs(math.sin(self.sim.pose[3])))
        penalty *= (1. - abs(math.sin(self.sim.pose[4])))
        penalty *= (1. - abs(math.sin(self.sim.pose[5])))

        delta = abs(self.sim.pose[:3] - self.target_pos)
        r = math.sqrt(np.dot(delta, delta))
        
        if(r > 0.01): decay = math.exp(-1/r) 
        else: decay = 0
        reward = 1. - decay
        reward *= penalty
        return reward

    def getPose(self):
        return self.sim.pose

    def step(self, rotor_speeds):

        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) 
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
    
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
