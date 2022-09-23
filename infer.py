import os, sys
import numpy as np
import trimesh
import torch
from networks import IGR, lbs_mlp, learnt_representations
from smpl_pytorch.smpl_server import SMPLServer
from utils.deform import rotate_root_pose_x, infer


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

''' dir to dump mesh '''
output_folder = './output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

''' Load pretrained models and necessary files '''
data = np.load('extra-data/shapedirs_f.npz')
shapedirs = torch.FloatTensor(data['shapedirs']).cuda()
tfs_weighted_zero = torch.FloatTensor(data['tfs_weighted_zero']).cuda()
lbs_weights = torch.FloatTensor(data['lbs_weights']).cuda()
num_v = len(shapedirs)
model_blend_weight = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=num_v, width=512, depth=8, skip_layer=[4]).cuda().eval()
model_blend_weight.load_state_dict(torch.load('extra-data/pretrained/blend_weight.pth'))

numG = 100
dim_latentG = 12

model_G_shirt = IGR.ImplicitNet_multiG(d_in=3+dim_latentG, skip_in=[4]).cuda().eval()
model_G_pants = IGR.ImplicitNet_multiG(d_in=3+dim_latentG, skip_in=[4]).cuda().eval()
model_G_shirt.load_state_dict(torch.load('extra-data/pretrained/shirt.pth'))
model_G_pants.load_state_dict(torch.load('extra-data/pretrained/pants.pth'))
model_G_shirt = model_G_shirt.cuda().eval()
model_G_pants = model_G_pants.cuda().eval()

model_rep_shirt = learnt_representations.Network(cloth_rep_size=dim_latentG, samples=numG)
model_rep_pants = learnt_representations.Network(cloth_rep_size=dim_latentG, samples=numG)
model_rep_shirt.load_state_dict(torch.load('extra-data/pretrained/shirt_rep.pth'))
model_rep_pants.load_state_dict(torch.load('extra-data/pretrained/pants_rep.pth'))
model_rep_shirt = model_rep_shirt.cuda().eval()
model_rep_pants = model_rep_pants.cuda().eval()
print('Load SDF model done!')

embedder, embed_dim = lbs_mlp.get_embedder(4)
d_width = 512
dim_theta = 72
dim_theta_p = 128 
model_lbs = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=24, width=d_width, depth=8, skip_layer=[4])
model_lbs.load_state_dict(torch.load('extra-data/pretrained/lbs_shirt.pth'))
model_lbs = model_lbs.cuda().eval()
model_lbs_p = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=24, width=d_width, depth=8, skip_layer=[4])
model_lbs_p.load_state_dict(torch.load('extra-data/pretrained/lbs_pants.pth'))
model_lbs_p = model_lbs_p.cuda().eval()
print('Load lbs model done!')
model_lbs_delta = lbs_mlp.lbs_pbs(d_in_theta=dim_theta, d_in_x=embed_dim, d_out_p=dim_theta_p, skip=True, hidden_theta=d_width, hidden_matrix=d_width)
model_lbs_delta = model_lbs_delta.cuda().eval()
model_lbs_delta.load_state_dict(torch.load('extra-data/pretrained/lbs_delta_shirt.pth'))
model_lbs_delta_p = lbs_mlp.lbs_pbs(d_in_theta=dim_theta, d_in_x=embed_dim, d_out_p=dim_theta_p, skip=True, hidden_theta=d_width, hidden_matrix=d_width)
model_lbs_delta_p = model_lbs_delta_p.cuda().eval()
model_lbs_delta_p.load_state_dict(torch.load('extra-data/pretrained/lbs_delta_pants.pth'))
print('Load lbs_delta model done!')

''' Initialize SMPL model '''
rest_pose = np.zeros((24,3), np.float32)
rest_pose[1,2] = 0.15
rest_pose[2,2] = -0.15
rest_pose = rotate_root_pose_x(rest_pose)

param_canonical = torch.zeros((1, 86),dtype=torch.float32).cuda()
param_canonical[0, 0] = 1
param_canonical[:,4:76] = torch.FloatTensor(rest_pose).reshape(-1)
smpl_server = SMPLServer(param_canonical, gender='f', betas=None, v_template=None)
tfs_c_inv = smpl_server.tfs_c_inv.detach()

''' Initialize SMPL/garment parameters '''
body_parameter = torch.load('extra-data/pose-beta-sample.pt')
pose = body_parameter[:,:72].cuda()
beta = body_parameter[:,72:].cuda()
z_style_shirt = model_rep_shirt.weights[0] # these weights are pretrained latent code for the 100 shirts
z_style_pants = model_rep_pants.weights[0] # these weights are pretrained latent code for the 100 pants

''' Reconstruct and deform garments '''
body_mesh, shirt_mesh, pants_mesh, shirt_mesh_T, pants_mesh_T = infer(pose, beta, model_G_shirt, model_G_pants, z_style_shirt, z_style_pants, tfs_c_inv, shapedirs, tfs_weighted_zero, embedder, model_lbs, model_lbs_delta, model_lbs_p, model_lbs_delta_p, model_blend_weight, smpl_server, output_folder)
body_mesh.export(output_folder + '/body.obj')
shirt_mesh.export(output_folder + '/shirt.obj')
pants_mesh.export(output_folder + '/pants.obj')
shirt_mesh_T.export(output_folder + '/shirt_T.obj')
pants_mesh_T.export(output_folder + '/pants_T.obj')