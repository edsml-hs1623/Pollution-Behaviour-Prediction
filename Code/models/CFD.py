#!/usr/bin/env python

#  Copyright (C) 2023
#  
#  Boyang Chen, Claire Heaney, Christopher Pain
#  Applied Modelling and Computation Group
#  Department of Earth Science and Engineering
#  Imperial College London
#
#  boyang.chen16@imperial.ac.uk
#  
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation,
#  version 3.0 of the License.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.

#-- Import general libraries
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' ## enable xla devices # Comment out this line if runing on GPU cluster
import numpy as np 
import pandas as pd
import time 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check if GPU is available
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
# print("Running on GPU: ", is_gpu)

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
nx = 512
ny = 512
nz = 64
dx = 1.0 ; dy = 1.0 ; dz = 1.0
Re = 0.15
dt = 0.5
ub = -1.0
# ratio = int(max(nx, ny, nz) / min(nx, ny, nz))
ratio_x = int(nx/nz)
ratio_y = int(ny/nz)
nlevel = int(math.log(min(nx, ny, nz), 2)) + 1
# print('How many levels in multigrid:', nlevel)
# print('Aspect ratio:', ratio_x)
# print('Grid spacing:', dx)
# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0])
# Laplacian filters
pd1 = torch.tensor([[2/26, 3/26, 2/26],
       [3/26, 6/26, 3/26],
       [2/26, 3/26, 2/26]])
pd2 = torch.tensor([[3/26, 6/26, 3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26, 3/26]])
pd3 = torch.tensor([[2/26, 3/26, 2/26],
       [3/26, 6/26, 3/26],
       [2/26, 3/26, 2/26]])
w1 = torch.zeros([1, 1, 3, 3, 3])
wA = torch.zeros([1, 1, 3, 3, 3])
w1[0, 0, 0,:,:] = pd1/dx**2
w1[0, 0, 1,:,:] = pd2/dx**2
w1[0, 0, 2,:,:] = pd3/dx**2
wA[0, 0, 0,:,:] = -pd1/dx**2
wA[0, 0, 1,:,:] = -pd2/dx**2
wA[0, 0, 2,:,:] = -pd3/dx**2
# Gradient filters
p_div_x1 = torch.tensor([[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]])
p_div_x2 = torch.tensor([[-0.056, 0.0, 0.056],
       [-0.22, 0.0, 0.22],
       [-0.056, 0.0, 0.056]])
p_div_x3 = torch.tensor([[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]])
p_div_y1 = torch.tensor([[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]])
p_div_y2 = torch.tensor([[0.056, 0.22, 0.056],
       [0.0, 0.0, 0.0],
       [-0.056, -0.22, -0.056]])
p_div_y3 = torch.tensor([[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]])
p_div_z1 = torch.tensor([[0.014, 0.056, 0.014],
       [0.056, 0.22, 0.056],
       [0.014, 0.056, 0.014]])
p_div_z2 = torch.tensor([[0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0]])
p_div_z3 = torch.tensor([[-0.014, -0.056, -0.014],
       [-0.056, -0.22, -0.056],
       [-0.014, -0.056, -0.014]])
w2 = torch.zeros([1,1,3,3,3])
w3 = torch.zeros([1,1,3,3,3])
w4 = torch.zeros([1,1,3,3,3])
w2[0,0,0,:,:] = -p_div_x1/dx*0.5
w2[0,0,1,:,:] = -p_div_x2/dx*0.5
w2[0,0,2,:,:] = -p_div_x3/dx*0.5
w3[0,0,0,:,:] = -p_div_y1/dx*0.5
w3[0,0,1,:,:] = -p_div_y2/dx*0.5
w3[0,0,2,:,:] = -p_div_y3/dx*0.5
w4[0,0,0,:,:] = -p_div_z1/dx*0.5
w4[0,0,1,:,:] = -p_div_z2/dx*0.5
w4[0,0,2,:,:] = -p_div_z3/dx*0.5
# Restriction filters
w_res = torch.zeros([1,1,2,2,2])
w_res[0,0,:,:,:] = 0.125
################# Numerical parameters ################
ntime = 2000                      # Time steps
n_out = 2000                      # Results output
iteration = 5                     # Multigrid iteration
nrestart = 0                      # Last time step for restart
ctime_old = 0                     # Last ctime for restart
LIBM = True                       # Immersed boundary method
ctime = 0                         # Initialise ctime
save_fig = True                   # Save results
Restart = False                   # Restart
eplsion_k = 1e-04                 # Stablisatin factor in Petrov-Galerkin for velocity
diag = np.array(wA)[0,0,1,1,1]    # Diagonal component
#######################################################
# # # ################################### # # #
# # # ######    Create tensor      ###### # # #
# # # ################################### # # #
input_shape = (1,1,nz,ny,nx)
values_u = torch.zeros(input_shape, device=device, requires_grad=True)
values_v = torch.zeros(input_shape, device=device, requires_grad=True)
values_w = torch.zeros(input_shape, device=device, requires_grad=True)
values_p = torch.zeros(input_shape, device=device, requires_grad=True)
k1 = torch.ones(input_shape, device=device)*2.0
input_shape_pad = (1,1,nz+2,ny+2,nx+2)
values_uu = torch.zeros(input_shape_pad, device=device, requires_grad=True)
values_vv = torch.zeros(input_shape_pad, device=device, requires_grad=True)
values_ww = torch.zeros(input_shape_pad, device=device, requires_grad=True)
values_pp = torch.zeros(input_shape_pad, device=device, requires_grad=True)
b_uu = torch.zeros(input_shape_pad, device=device)
b_vv = torch.zeros(input_shape_pad, device=device)
b_ww = torch.zeros(input_shape_pad, device=device)
k_uu = torch.zeros(input_shape_pad, device=device)
k_vv = torch.zeros(input_shape_pad, device=device)
k_ww = torch.zeros(input_shape_pad, device=device)
# #######################################################    
# ################# Only for IBM ########################
if LIBM == True:
    mesh = np.load("Urban_street.npy")
    sigma = torch.zeros(input_shape, dtype=torch.float32, device=device)
    for i in range(nz):
        sigma[0,0,i,:,:] = torch.tensor(mesh[0,:,:,i,0])
    sigma = sigma.transpose_(4, 3)
    sigma = torch.flip(sigma, [3])
    sigma = torch.where(sigma == 0, torch.tensor(1e08, dtype=torch.float32, device=device), torch.tensor(0, dtype=torch.float32, device=device))
#######################################################
# # # ################################### # # #
# # # #########  AI4Urban MAIN ########## # # #
# # # ################################### # # #
class AI4Urban(nn.Module):
    """docstring for AI4Urban"""
    def __init__(self):
        super(AI4Urban, self).__init__()
        # self.arg = arg
        self.xadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.zadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

        self.A = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.res = nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)

        self.A.weight.data = wA
        self.res.weight.data = w_res
        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.zadv.weight.data = w4

        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.zadv.bias.data = bias_initializer
###############################################################
    def boundary_condition_u(self, values_u, values_uu):
        nz = values_u.shape[2]
        ny = values_u.shape[3]
        nx = values_u.shape[4]
        nnz = values_uu.shape[2]
        nny = values_uu.shape[3]
        nnx = values_uu.shape[4]

        # Create a copy to avoid in-place operations
        values_uu_new = values_uu.clone()
        values_u_new = values_u.clone()
        values_uu_new[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_u_new[0,0,:,:,:]        

        values_uu_new[0,0,:,:,0].fill_(ub)
        values_uu_new[0,0,:,:,nx+1].fill_(ub)
        values_uu_new[0,0,:,0,:] = values_uu_new[0,0,:,1,:]
        values_uu_new[0,0,:,ny+1,:] = values_uu_new[0,0,:,ny,:]
        values_uu_new[0,0,0,:,:] = values_uu_new[0,0,1,:,:]*0 
        values_uu_new[0,0,nz+1,:,:] = values_uu_new[0,0,nz,:,:]
        return values_uu_new

    def boundary_condition_v(self, values_v, values_vv):
        nz = values_v.shape[2]
        ny = values_v.shape[3]
        nx = values_v.shape[4]
        nnz = values_vv.shape[2]
        nny = values_vv.shape[3]
        nnx = values_vv.shape[4]

        values_vv_new = values_vv.clone()
        values_v_new = values_v.clone()
        values_vv_new[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_v_new[0,0,:,:,:]

        values_vv_new[0,0,:,:,0] = values_vv_new[0,0,:,:,1]*0 
        values_vv_new[0,0,:,:,nx+1] = values_vv_new[0,0,:,:,nx]*0
        values_vv_new[0,0,:,0,:].fill_(0.0)
        values_vv_new[0,0,:,ny+1,:].fill_(0.0)
        values_vv_new[0,0,0,:,:] = values_vv_new[0,0,1,:,:]*0
        values_vv_new[0,0,nz+1,:,:] = values_vv_new[0,0,nz,:,:]
        return values_vv_new

    def boundary_condition_w(self, values_w, values_ww):
        nz = values_w.shape[2]
        ny = values_w.shape[3]
        nx = values_w.shape[4]
        nnz = values_ww.shape[2]
        nny = values_ww.shape[3]
        nnx = values_ww.shape[4]

        # Create a copy to avoid in-place operations
        values_ww_new = values_ww.clone()
        values_w_new = values_w.clone()
        values_ww_new[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_w_new[0,0,:,:,:]
        
        values_ww_new[0,0,:,:,0] =  values_ww_new[0,0,:,:,1]*0 
        values_ww_new[0,0,:,:,nx+1] = values_ww_new[0,0,:,:,nx]*0
        values_ww_new[0,0,:,0,:] = values_ww_new[0,0,:,1,:]*0
        values_ww_new[0,0,:,ny+1,:] = values_ww_new[0,0,:,ny,:]*0
        values_ww_new[0,0,0,:,:].fill_(0.0)
        values_ww_new[0,0,nz+1,:,:].fill_(0.0)
        return values_ww_new

    def solid_body(self, values_u, values_v, values_w, sigma, dt):
        
        # print(f"in solid body fun, values_u shape: {values_u.shape}, sigma shape = {sigma.shape}, dt: {dt}")
        values_u_new, values_v_new, values_w_new = values_u.clone(), values_v.clone(), values_w.clone()
        values_u_new = values_u / (1+dt*sigma)
        values_v_new = values_v / (1+dt*sigma)
        values_w_new = values_w / (1+dt*sigma)

        return values_u_new, values_v_new, values_w_new

    def boundary_condition_p(self, values_p, values_pp):  # need check this boundary condition for the real case 

        # Check the shapes of values_p and values_pp
        assert values_p.ndimension() == 5, f"Expected values_p to have 5 dimensions, but got {values_p.ndimension()} dimensions with shape {values_p.shape}"
        assert values_pp.ndimension() == 5, f"Expected values_pp to have 5 dimensions, but got {values_pp.ndimension()} dimensions with shape {values_pp.shape}"

        nz = values_p.shape[2]
        ny = values_p.shape[3]
        nx = values_p.shape[4]
        nnz = values_pp.shape[2]
        nny = values_pp.shape[3]
        nnx = values_pp.shape[4]

        values_pp_new = values_pp.clone()
        values_p_new = values_p.clone()

        values_pp_new[0,0,1:nnz-1,1:nny-1,1:nnx-1] = values_p_new[0,0,:,:,:]
        values_pp_new[0,0,:,:,0] =  values_pp_new[0,0,:,:,1] 
        values_pp_new[0,0,:,:,nx+1] = values_pp_new[0,0,:,:,nx] * 0 # outflow boundary condition 
        values_pp_new[0,0,:,0,:] = values_pp_new[0,0,:,1,:]
        values_pp_new[0,0,:,ny+1,:] = values_pp_new[0,0,:,ny,:]
        values_pp_new[0,0,0,:,:] = values_pp_new[0,0,1,:,:]
        values_pp_new[0,0,nz+1,:,:] = values_pp_new[0,0,nz,:,:]
        return values_pp_new

    def boundary_condition_k(self, k_u, k_uu):
        nz = k_u.shape[2]
        ny = k_u.shape[3]
        nx = k_u.shape[4]
        nnz = k_uu.shape[2]
        nny = k_uu.shape[3]
        nnx = k_uu.shape[4]

        k_uu_new = k_uu.clone()
        k_u_new = k_u.clone()

        k_uu_new[0,0,1:nnz-1,1:nny-1,1:nnx-1] = k_u_new[0,0,:,:,:]
        k_uu_new[0,0,:,:,0] =  k_uu_new[0,0,:,:,1]*0 
        k_uu_new[0,0,:,:,nx+1] = k_uu_new[0,0,:,:,nx]*0 
        k_uu_new[0,0,:,0,:] = k_uu_new[0,0,:,1,:]*0
        k_uu_new[0,0,:,ny+1,:] = k_uu_new[0,0,:,ny,:]*0
        k_uu_new[0,0,0,:,:] = k_uu_new[0,0,1,:,:]*0
        k_uu_new[0,0,nz+1,:,:] = k_uu_new[0,0,nz,:,:]*0

        return k_uu_new

    def boundary_condition_cw(self, w):
        nz = w.shape[2]
        ny = w.shape[3]
        nx = w.shape[4]
        ww = F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        ww_new = ww.clone()

        ww_new[0,0,:,:,0] =  ww_new[0,0,:,:,1]*0
        ww_new[0,0,:,:,nx+1] = ww_new[0,0,:,:,nx]*0 
        ww_new[0,0,:,0,:] = ww_new[0,0,:,1,:]*0
        ww_new[0,0,:,ny+1,:] = ww_new[0,0,:,ny,:]*0
        ww_new[0,0,0,:,:] = ww_new[0,0,1,:,:]*0
        ww_new[0,0,nz+1,:,:] = ww_new[0,0,nz,:,:]*0
        return ww_new

    def F_cycle_MG(self, values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, ratio_x, ratio_y):
        b = -(self.xadv(values_uu) + self.yadv(values_vv) + self.zadv(values_ww)) / dt

        # Initialize w and r before the loop
        w = torch.zeros_like(values_p)
        r = torch.zeros_like(values_p)

        for MG in range(iteration):
              w = torch.zeros((1,1,2,2*ratio_y,2*ratio_x), device=device).detach()
              ww = torch.zeros((1,1,2+2,2*ratio_y+2,2*ratio_x+2), device=device).detach()
              r = self.A(self.boundary_condition_p(values_p, values_pp)) - b 
              r_s = []  
              r_s.append(r)
              for i in range(1,nlevel-1):
                     r = self.res(r)
                     r_s.append(r)
              for i in reversed(range(1, nlevel-1)):
                     ww = self.boundary_condition_cw(w)
                     w = w.clone() - self.A(ww) / diag + r_s[i] / diag
                     w = self.prol(w)        
              values_p = values_p.clone() - w
              values_p = values_p.clone() - self.A(self.boundary_condition_p(values_p.clone(), values_pp)) / diag + b / diag
        return values_p, w, r

    def PG_vector(self, values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w):
        k_u = 0.1 * dx * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_u) / \
            (1e-03 + (torch.abs(ADx_u) * dx**-3 + torch.abs(ADy_u) * dx**-3 + torch.abs(ADz_u) * dx**-3) / 3)

        k_v = 0.1 * dy * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_v) / \
            (1e-03 + (torch.abs(ADx_v) * dx**-3 + torch.abs(ADy_v) * dx**-3 + torch.abs(ADz_v) * dx**-3) / 3)

        k_w = 0.1 * dz * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_w) / \
            (1e-03 + (torch.abs(ADx_w) * dx**-3 + torch.abs(ADy_w) * dx**-3 + torch.abs(ADz_w) * dx**-3) / 3)

        k_u = torch.minimum(k_u.clone(), k1) / (1+dt*sigma) 
        k_v = torch.minimum(k_v.clone(), k1) / (1+dt*sigma) 
        k_w = torch.minimum(k_w.clone(), k1) / (1+dt*sigma) 

        k_uu = self.boundary_condition_k(k_u,k_uu.clone())     # **************************** halo update -> k_uu ****************************
        k_vv = self.boundary_condition_k(k_v,k_vv.clone())     # **************************** halo update -> k_vv ****************************
        k_ww = self.boundary_condition_k(k_w,k_ww.clone())     # **************************** halo update -> k_ww ****************************

        k_x = 0.5 * (k_u * AD2_u + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 0.5 * (k_v * AD2_v + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        k_z = 0.5 * (k_w * AD2_w + self.diff(values_ww * k_ww) - values_w * self.diff(k_ww))
        return k_x, k_y, k_z

    def forward(self, values_u, values_uu, values_v, values_vv, values_w, values_ww, values_p, values_pp, b_uu, b_vv, b_ww, k1, dt, iteration, k_uu, k_vv, k_ww):
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u.clone(), values_v.clone(), values_w.clone(), sigma, dt)
    # Padding velocity vectors 
        values_uu = self.boundary_condition_u(values_u,values_uu.clone()) # ****************************** halo update -> values_uu ************************  
        values_vv = self.boundary_condition_v(values_v,values_vv.clone()) # ****************************** halo update -> values_vv ************************ 
        values_ww = self.boundary_condition_w(values_w,values_ww.clone()) # ****************************** halo update -> values_ww ************************ 
        values_pp = self.boundary_condition_p(values_p,values_pp.clone()) # ****************************** halo update -> values_pp ************************ 

        Grapx_p = self.xadv(values_pp) * dt ; Grapy_p = self.yadv(values_pp) * dt ; Grapz_p = self.zadv(values_pp) * dt  
        ADx_u = self.xadv(values_uu) ; ADy_u = self.yadv(values_uu) ; ADz_u = self.zadv(values_uu)
        ADx_v = self.xadv(values_vv) ; ADy_v = self.yadv(values_vv) ; ADz_v = self.zadv(values_vv)
        ADx_w = self.xadv(values_ww) ; ADy_w = self.yadv(values_ww) ; ADz_w = self.zadv(values_ww)
        AD2_u = self.diff(values_uu) ; AD2_v = self.diff(values_vv) ; AD2_w = self.diff(values_ww)
    # First step for solving uvw
        [k_x,k_y,k_z] = self.PG_vector(values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, 
                                       ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w)

        b_u = values_u + 0.5 * (Re * k_x * dt - values_u * ADx_u * dt - values_v * ADy_u * dt - values_w * ADz_u * dt) - Grapx_p
        b_v = values_v + 0.5 * (Re * k_y * dt - values_u * ADx_v * dt - values_v * ADy_v * dt - values_w * ADz_v * dt) - Grapy_p
        b_w = values_w + 0.5 * (Re * k_z * dt - values_u * ADx_w * dt - values_v * ADy_w * dt - values_w * ADz_w * dt) - Grapz_p
    # Solid body
        if LIBM == True: [b_u, b_v, b_w] = self.solid_body(b_u.clone(), b_v.clone(), b_w.clone(), sigma, dt)
    # Padding velocity vectors 
        b_uu = self.boundary_condition_u(b_u,b_uu.clone()) # ****************************** halo update -> b_uu ************************ 
        b_vv = self.boundary_condition_v(b_v,b_vv.clone()) # ****************************** halo update -> b_vv ************************ 
        b_ww = self.boundary_condition_w(b_w,b_ww.clone()) # ****************************** halo update -> b_ww ************************ 

        ADx_u = self.xadv(b_uu) ; ADy_u = self.yadv(b_uu) ; ADz_u = self.zadv(b_uu)
        ADx_v = self.xadv(b_vv) ; ADy_v = self.yadv(b_vv) ; ADz_v = self.zadv(b_vv)
        ADx_w = self.xadv(b_ww) ; ADy_w = self.yadv(b_ww) ; ADz_w = self.zadv(b_ww)
        AD2_u = self.diff(b_uu) ; AD2_v = self.diff(b_vv) ; AD2_w = self.diff(b_ww)

        [k_x,k_y,k_z] = self.PG_vector(b_uu, b_vv, b_ww, b_u, b_v, b_w, k1, k_uu, k_vv, k_ww, 
                                       ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w)   
    # Second step for solving uvw   
        values_u = values_u.clone() + Re * k_x * dt - b_u * ADx_u * dt - b_v * ADy_u * dt - b_w * ADz_u * dt - Grapx_p  
        values_v = values_v.clone() + Re * k_y * dt - b_u * ADx_v * dt - b_v * ADy_v * dt - b_w * ADz_v * dt - Grapy_p  
        values_w = values_w.clone() + Re * k_z * dt - b_u * ADx_w * dt - b_v * ADy_w * dt - b_w * ADz_w * dt - Grapz_p
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u.clone(), values_v.clone(), values_w.clone(), sigma, dt)
    # pressure
        values_uu = self.boundary_condition_u(values_u,values_uu.clone()) # ****************************** halo update -> values_uu ************************ 
        values_vv = self.boundary_condition_v(values_v,values_vv.clone()) # ****************************** halo update -> values_vv ************************ 
        values_ww = self.boundary_condition_w(values_w,values_ww.clone()) # ****************************** halo update -> values_ww ************************
        [values_p, w ,r] = self.F_cycle_MG(values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, ratio_x, ratio_y)
    # Pressure gradient correction    
        values_pp = self.boundary_condition_p(values_p, values_pp.clone()) # ****************************** halo update -> values_pp ************************       
        values_u = values_u.clone() - self.xadv(values_pp) * dt ; values_v = values_v - self.yadv(values_pp) * dt ; values_w = values_w - self.zadv(values_pp) * dt      
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u.clone(), values_v.clone(), values_w.clone(), sigma, dt)
        return values_u, values_v, values_w, values_p, w, r
