import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from skimage import measure
from utils.sdf import create_grid, eval_grid_octree, eval_grid

def rotate_root_pose_x(pose):
    rot_x_90 = torch.FloatTensor([[1, 0, 0, 0],
                                [0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]]).unsqueeze(0)

    root_pose = angle_axis_to_rotation_matrix(torch.Tensor(pose[0]).unsqueeze(0))
    root_pose_rot = rot_x_90.matmul(root_pose)
    root_pose_rot = root_pose_rot[:,:3,:]
    root_pose_rot[:,-1,-1] = 1
    pose[0] = rotation_matrix_to_angle_axis(root_pose_rot[:,:3,:]).squeeze().numpy()
    return pose

def reconstruct(cloth_representation, model_G, resolution=256, thresh=0, just_vf=False, is_trousers=False):
    if is_trousers:
        b_min = np.array([-0.4, -0.6, -0.95])
        b_max = np.array([0.4, 0.6, 0.4])
    else:
        b_min = np.array([-0.8, -0.6, -0.5])
        b_max = np.array([0.8, 0.6, 0.8])

    coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)

    def eval_func_xyz(points):
        points = torch.FloatTensor(points).transpose(1, 0).cuda()
        num_points = len(points)
        x_cloth_points = cloth_representation.unsqueeze(0).repeat(num_points, 1)
        pred = model_G(points, x_cloth_points, num_points)*-100
        pred = pred.reshape(1,-1)
        return pred.cpu().data.numpy()

    sdf = eval_grid_octree(coords, eval_func_xyz, threshold = 0.01, num_samples=10000, init_resolution=16)

    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thresh)
    except AttributeError:
        verts, faces, normals, values = measure.marching_cubes(sdf, thresh)

    cloth_mesh = trimesh.Trimesh(np.float64(verts), faces[:, ::-1])
    cloth_mesh.vertices /= resolution
    cloth_mesh.vertices *= (b_max - b_min)[None]
    cloth_mesh.vertices += b_min[None]

    if just_vf:
        return cloth_mesh.vertices, cloth_mesh.faces 

    return cloth_mesh

def skinning(x, delta_theta, delta_beta, w, tfs, tfs_c_inv):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        tfs_c_inv (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    tfs = torch.einsum('bnij,bnjk->bnik', tfs, tfs_c_inv)

    x_disp = x + delta_theta + delta_beta
    x_h = F.pad(x_disp, (0, 1), value=1.0)
    x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

    return x_h[:, :, :3]

def deform(vertices_garment_T, faces, smpl_tfs, tfs_c_inv, pose, beta, shapedirs, tfs_weighted_zero, embedder, model_lbs, model_lbs_delta, model_blend_weight):
    # vertices_garment_T - (#P, 3) 
    # smpl_tfs - (24, 4, 4) 
    # tfs_c_inv - (24, 4, 4) 
    # pose - (1, 72) 
    # beta - (1, 10) 

    vertices_garment_T = vertices_garment_T.unsqueeze(0)
    smpl_tfs = smpl_tfs.unsqueeze(0)
    tfs_c_inv = tfs_c_inv.unsqueeze(0)
    vertices_garment_T_embed = embedder(vertices_garment_T) 
    pose_repeat = pose.unsqueeze(0).repeat(1, vertices_garment_T.shape[1], 1)

    weights_points = model_blend_weight(vertices_garment_T, None) # [B, N, V]
    weights_points = weights_points.softmax(dim=-1)

    blend_shape = torch.einsum('bl,mkl->bmk', [beta, shapedirs])
    homogen_coord = torch.zeros(blend_shape.shape[0], blend_shape.shape[1], 1).cuda()
    blend_shape_homo = torch.cat([blend_shape, homogen_coord], dim=2)
    blend_shape_homo = torch.matmul(tfs_weighted_zero.unsqueeze(0), torch.unsqueeze(blend_shape_homo, dim=-1))
    blend_shape = blend_shape_homo[:, :, :3, 0]
    delta_beta = torch.einsum('bnv,bvk->bnk', [weights_points, blend_shape])

    lbs_weight = model_lbs(vertices_garment_T, None)
    lbs_weight = lbs_weight.softmax(dim=-1)
    delta_theta = model_lbs_delta(pose_repeat, vertices_garment_T_embed, None)

    verts_deformed = skinning(vertices_garment_T, delta_theta, delta_beta, lbs_weight, smpl_tfs, tfs_c_inv)
    verts_deformed = verts_deformed.squeeze() # (#P, 3) 
    verts_deformed_np = verts_deformed.detach().cpu().numpy()

    cloth_mesh = trimesh.Trimesh(verts_deformed_np, faces)
    return verts_deformed, cloth_mesh, delta_theta, delta_beta


def infer(pose, beta, model_G_shirt, model_G_pants, z_style_shirt, z_style_pants, tfs_c_inv, shapedirs, tfs_weighted_zero, embedder, model_lbs, model_lbs_delta, model_lbs_p, model_lbs_delta_p, model_blend_weight, smpl_server, output_folder):

    with torch.no_grad():
        shirt_mesh_T = reconstruct(z_style_shirt, model_G_shirt, just_vf=False, resolution=256)
        print('Reconstructing shirt done!')
        pants_mesh_T = reconstruct(z_style_pants, model_G_pants, just_vf=False, resolution=256, is_trousers=True)
        print('Reconstructing pants done!')
        shirt_verts, shirt_faces = shirt_mesh_T.vertices, shirt_mesh_T.faces 
        pants_verts, pants_faces = pants_mesh_T.vertices, pants_mesh_T.faces 


        shirt_verts = torch.from_numpy(shirt_verts).float().cuda()
        pants_verts = torch.from_numpy(pants_verts).float().cuda()

        smpl_params = torch.zeros((1, 86),dtype=torch.float32).cuda()
        smpl_params[0, 0] = 1
        smpl_params[:,4:76] = pose.reshape(-1)
        smpl_params[:,76:] = beta.reshape(-1)
        smpl_output = smpl_server(smpl_params, absolute=True)
        smpl_verts = smpl_output['smpl_verts'].squeeze().cpu().numpy()
        smpl_tfs = smpl_output['smpl_tfs'].squeeze()


        verts_shirt_deformed, shirt_mesh, delta_theta, delta_beta = deform(shirt_verts, shirt_faces, smpl_tfs, tfs_c_inv, pose, beta, shapedirs, tfs_weighted_zero, embedder, model_lbs, model_lbs_delta, model_blend_weight)
        verts_pants_deformed, pants_mesh, delta_theta_p, delta_beta_p = deform(pants_verts, pants_faces, smpl_tfs, tfs_c_inv, pose, beta, shapedirs, tfs_weighted_zero, embedder, model_lbs_p, model_lbs_delta_p, model_blend_weight)
        print('Skinning done!')

        smpl_faces = smpl_server.smpl.faces
        body_mesh = trimesh.Trimesh(smpl_verts, smpl_faces)
        colors_f_body = np.ones((len(smpl_faces), 4))*np.array([255, 255, 255, 200])[np.newaxis,:]
        colors_f_shirt = np.ones((len(shirt_faces), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
        colors_f_pants = np.ones((len(pants_faces), 4))*np.array([122, 122, 122, 200])[np.newaxis,:]
        body_mesh.visual.face_colors = colors_f_body
        shirt_mesh.visual.face_colors = colors_f_shirt
        pants_mesh.visual.face_colors = colors_f_pants
        shirt_mesh_T.visual.face_colors = colors_f_shirt
        pants_mesh_T.visual.face_colors = colors_f_pants
        
    return body_mesh, shirt_mesh, pants_mesh, shirt_mesh_T, pants_mesh_T