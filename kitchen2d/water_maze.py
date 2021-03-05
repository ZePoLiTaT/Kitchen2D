#!/usr/bin/env python
# Copyright (c) 2017 Zi Wang
import kitchen2d.kitchen_stuff as ks
from kitchen2d.kitchen_stuff import Kitchen2D, WaterMaze2D
from kitchen2d.gripper import Gripper, incup
from kitchen2d.kitchen_constants import *
import sys
import numpy as np
#import cPickle as pickle
import pickle as cPickle
import os
import time
settings = {
    0: {
        'do_gui': False,
        'sink_d': 1.,
        'sink_pos_x': -0.,
        'sink_pos_y': 20.,
        'left_table_width': 100.,
        'right_table_width': 100.,
        'faucet_h': 0.5,
        'faucet_w': 0.5,
        'planning': False,
        'save_fig': False,
        'liquid_frequency': 1.0,
        'overclock': 50 # number of frames to skip when showing graphics.
    }
}

class WaterMaze(object):
    def __init__(self, **kwargs):

        self.x_range = np.array(
            [[4., 4.,   500, -10.], 
             [5., 5.,  1000,  10.]])
        self.names = ['cw1', 'ch1', 'vol', 'liquid_x']
        #self.lengthscale_bound = np.array([np.ones(8)*0.1, [0.15, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5]])
        self.context_idx = [0, 1, 2, 3]
        self.param_idx = []

        # --- Support for n obstacles
        self.context_idx_start = self.x_range.shape[1]
        self.x_range_obstacle = np.array(
            [[-10., 1.  ], 
             [ 10., 10. ]])
        self.names_obs = ['rel_x', 'rel_y']
        self.n_obstacles = kwargs.get('n_obstacles', 1)
        # 1. Replicate n times obstacles params/indices/names
        x_range_obs_exp = np.tile(self.x_range_obstacle, self.n_obstacles)
        context_obs_idx = np.arange(self.context_idx_start, self.context_idx_start + x_range_obs_exp.shape[1])
        names_obs_exp = [['{}_{}'.format(name, i) for name in self.names_obs] for i in range(self.n_obstacles)]
        # 2. Append at the end of current param/indices/names
        self.x_range = np.hstack([self.x_range, x_range_obs_exp])
        self.param_idx.append(context_obs_idx)
        self.names.extend(np.array(names_obs_exp).flatten())


        self.dx = len(self.x_range[0])
        self.task_lengthscale = np.ones(8)*10
        self.do_gui = False
        self.sim_time = 200

        # for _ in range(self.n_obstacles):
        #     self.x_range = 
        
        
    def check_legal(self, x):
        
        cw1, ch1, vol, liquid_x = x[:self.context_idx_start]
        rel_obs = x[self.context_idx_start:]

        settings[0]['do_gui'] = self.do_gui
        settings[0]['sink_pos_x'] = liquid_x


        maze_env = WaterMaze2D(**settings[0])
        #maze_env.add_obstacles([((-15,3), (3,0.5)),])

        cup1 = ks.make_cup(maze_env, (0,0), 0, cw1, ch1, 0.5) 
        # Add n obstacles 
        for i in range(self.n_obstacles):
            rel_x, rel_y = rel_obs[i*2:(i+1)*2]
            maze_env.add_obstacles([((rel_x, ch1+rel_y), (-2, 0), (0, 2), (2, 0)),])
        
        self.pouring_tsteps = int(vol)
        self.maze_env = maze_env
        self.cup1 = cup1
        return True


    def sampled_x(self, n):
        i = 0
        while i < n:
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            legal = self.check_legal(x)
            if legal:
                i += 1
                yield x


    def __call__(self, x, image_name=None):

        if not self.check_legal(x):
            return -1.

        #water_pos = self.cup2.position - self.cup2.shift
        #self.maze_env.gen_liquid(water_pos, self.cup2.usr_w, self.cup2.usr_d , int(vol))
        #self.gripper.get_liquid_from_faucet(5)
        for t in range(self.pouring_tsteps):
            self.maze_env.enabled_faucet = True
            self.maze_env.step()

        for t in range(self.sim_time):
            self.maze_env.enabled_faucet = False
            self.maze_env.step()

        #success, score = self.gripper.pour(self.cup1, (rel_x, rel_y), dangle, exact_control=False, p_range=cw1/2)
        #print('Pouring ', x, success, np.exp(2*(score*10 - 9.5)) - 1.)

        liquid_particles = self.maze_env.liquid.particles
        in_to_cup, _, _ = incup(self.cup1, liquid_particles, p_range=SCREEN_WIDTH)
        score = len(in_to_cup) * 1.0 / len(liquid_particles)

        return np.exp(2*(score*10 - 9.5)) - 1.
        


if __name__ == '__main__':
    func = WaterMaze()
    N = 10
    samples = func.sampled_x(N)
    x = list(samples)
    for xx in x:
        start = time.time()
        print(func(xx))
        print(time.time() - start)
    