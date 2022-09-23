import os, sys
import numpy as np
import pyrender
import pickle as pkl
import random
import torch
import torch.nn.functional as F
import trimesh
from torchgeometry import angle_axis_to_rotation_matrix
import kaolin
from collections import OrderedDict
from .models import BaseModel
from networks import lbs_mlp, IGR

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self._name = 'skinning'

        self.parallel = opt.parallel
        self.d_in = 3
        self.skip_layer = [0]
        self.d_width = 512
        self.dim_theta = 72
        self.dim_theta_p = 128

        self.is_trousers = opt.is_trousers

        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()
        
        if self.parallel:
            self._init_parallel()

        # init
        self._init_losses()
        self.trim_mesh = None

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _init_create_networks(self):
        # generator network
        self.embedder, self.embed_dim = lbs_mlp.get_embedder(4)
        self._lbs = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=24, width=self.d_width, depth=8, skip_layer=[4])
        print(self._lbs)

        self._lbs_delta = lbs_mlp.lbs_pbs(d_in_theta=self.dim_theta, d_in_x=self.embed_dim, d_out_p=self.dim_theta_p, skip=True, hidden_theta=self.d_width, hidden_matrix=self.d_width)
        print(self._lbs_delta)

        if torch.cuda.is_available():
            self._lbs.cuda()
            self._lbs_delta.cuda()

        if self._is_train:
            self._body_sdf = IGR.ImplicitNet_multiG(d_in=3, skip_in=[2], dims=[256, 256, 256, 256]).cuda()
            self._body_sdf.load_state_dict(torch.load('extra-data/pretrained/sdf_body_f.pth'))
            for param in self._body_sdf.parameters():
                param.requires_grad = False
            print(self._body_sdf)

        data = np.load('extra_data/shapedirs_f.npz')
        self.shapedirs = torch.FloatTensor(data['shapedirs']).cuda()
        self.tfs_weighted_zero = torch.FloatTensor(data['tfs_weighted_zero']).cuda()
        self.lbs_weights = torch.FloatTensor(data['lbs_weights']).cuda()

        self.num_v = len(self.shapedirs)
        self._blend_weight = lbs_mlp.lbs_mlp_scanimate(d_in=3, d_out=self.num_v, width=self.d_width, depth=8, skip_layer=[4]).cuda().eval()
        self._blend_weight.load_state_dict(torch.load('extra-data/pretrained/blend_weight.pth'))

        print(self._blend_weight)
        for param in self._blend_weight.parameters():
            param.requires_grad = False

    def _init_parallel(self,):
        print('------------------------parallel------------------------')
        self._lbs = torch.nn.DataParallel(self._lbs)
        self._lbs_delta = torch.nn.DataParallel(self._lbs_delta)
        if self._is_train:
            self._body_sdf = torch.nn.DataParallel(self._body_sdf)

        self._blend_weight = torch.nn.DataParallel(self._blend_weight)

    def _init_train_vars(self):
        self._current_lr_lbs = self._opt.lr_G

        # initialize optimizers
        self._optimizer_lbs = torch.optim.Adam(list(self._lbs.parameters()) + list(self._lbs_delta.parameters()), lr=self._current_lr_lbs)

    def set_epoch(self, epoch):
        self.iepoch = epoch

    def _init_losses(self):
        self.record_loss_surface = []
        self.record_loss_around = []
        self.record_loss_collision = []
        self.record_loss_collision_order = []
        self.record_loss_collision_neutral = []
        self.record_loss_kl = []

    def set_input(self, input):
        self._input_index_cloth = input['index_cloth']
        self._input_surface_points = input['surface_points'].float() # (B, #frame, #P, 3)
        self._input_delta_beta = input['delta_beta'].float() # (B, #frame, #P, 3)
        self._input_random_points = input['random_points'].float() # (B, #frame, #P, 3)
        self._input_delta_beta_random = input['delta_beta_random'].float() # (B, #frame, #P, 3)
        self._input_closest_points = input['closest_points'].float() # (B, #frame, #P, 3)
        self._input_delta_beta_closest = input['delta_beta_closest'].float()
        self._input_smpl_tfs = input['smpl_tfs'].float() # (B, #frame, 24, 4, 4)

        self._input_tfs_c_inv = input['tfs_c_inv'].float() # (B, 24, 4, 4)
        self._deformed_points = input['deformed_points'] # (B, #frame, #P, 3)
        self._deformed_closest_points = input['deformed_closest_points'] 
        
        self._beta = input['beta']
        self._poses = input['poses']
        
        self._weight_surface = input['weight_surface']
        self._weight_random = input['weight_random']
        
        self._order_pair = input['order_pair']
        
        self._valid_pair_surface = input['valid_pair_surface_id']
        self._valid_pair_random = input['valid_pair_random_id']
        

        if torch.cuda.is_available():
            self._input_index_cloth = self._input_index_cloth.cuda()
            self._input_surface_points = self._input_surface_points.cuda()
            self._input_delta_beta = self._input_delta_beta.cuda()
            self._deformed_points = self._deformed_points.cuda()

            self._input_random_points = self._input_random_points.cuda()
            self._input_delta_beta_random = self._input_delta_beta_random.cuda()

            self._input_closest_points = self._input_closest_points.cuda()
            self._input_delta_beta_closest = self._input_delta_beta_closest.cuda()
            self._deformed_closest_points = self._deformed_closest_points.cuda()

            self._input_smpl_tfs = self._input_smpl_tfs.cuda()
            self._input_tfs_c_inv = self._input_tfs_c_inv.cuda()
            self._beta = self._beta.cuda()
            self._poses = self._poses.cuda()
            
            self._weight_surface = self._weight_surface.cuda()
            self._weight_random = self._weight_random.cuda()

            self._order_pair = self._order_pair.cuda()
            self._valid_pair_surface = self._valid_pair_surface.cuda()
            self._valid_pair_random = self._valid_pair_random.cuda()

        self._vertices_body = input['vertices_body'].cuda()

        self._faces_body = input['faces_body'].cuda()
        self._vertices_body_zero = input['vertices_body_zero'].cuda()
        
        self._B = self._input_surface_points.size(0)
        self._numPoint = self._input_surface_points.size(1)

        return 

    def set_train(self):
        self._lbs.train()
        self._lbs_delta.train()
        self._body_sdf.eval()
        self._is_train = True

    def set_eval(self):
        self._lbs.eval()
        self._lbs_delta.eval()
        self._is_train = False

    def reconstruct(self, vertices_garment_T, faces, smpl_tfs, pose=None, return_delta_theta=False):
        # vertices_garment_T - (#P, 3) 
        # smpl_tfs - (24, 4, 4) 
        with torch.no_grad():
            vertices_garment_T = vertices_garment_T.unsqueeze(0)
            smpl_tfs = smpl_tfs.unsqueeze(0)
            vertices_garment_T_embed = self.embedder(vertices_garment_T) 

            weights_points = self._blend_weight(vertices_garment_T, None) # [B, N, V]
            weights_points = weights_points.softmax(dim=-1)

            blend_shape = torch.einsum('bl,mkl->bmk', [self._beta[[0]], self.shapedirs])
            homogen_coord = torch.zeros(blend_shape.shape[0], blend_shape.shape[1], 1).cuda()
            blend_shape_homo = torch.cat([blend_shape, homogen_coord], dim=2)
            tfs_weighted_zero = self.tfs_weighted_zero.unsqueeze(0)
            blend_shape_homo = torch.matmul(tfs_weighted_zero, torch.unsqueeze(blend_shape_homo, dim=-1))
            blend_shape = blend_shape_homo[:, :, :3, 0]
            delta_beta = torch.einsum('bnv,bvk->bnk', [weights_points, blend_shape])

            pose = pose.unsqueeze(0).unsqueeze(1)
            pose = pose.repeat(1, vertices_garment_T.shape[1], 1)
            
            lbs_weight = self._lbs(vertices_garment_T)
            lbs_weight = lbs_weight.softmax(dim=-1)
            delta_theta = self._lbs_delta(pose, vertices_garment_T_embed)

            verts_deformed = self.skinning(vertices_garment_T, delta_theta, delta_beta, lbs_weight, smpl_tfs, self._input_tfs_c_inv[[0]])
            verts_deformed = verts_deformed.squeeze().cpu().numpy()

        cloth_mesh = trimesh.Trimesh(verts_deformed, faces)
        self.trim_mesh = cloth_mesh
        if return_delta_theta:
            return self.trim_mesh, delta_theta, delta_beta, lbs_weight, smpl_tfs, self._input_tfs_c_inv[[0]]
        else:
            return self.trim_mesh

    def optimize_parameters(self):
        if self._is_train:
            # convert tensor to variables
            self._optimizer_lbs.zero_grad()
            loss_lbs = self._forward_lbs()
            loss_lbs.backward()
            self._optimizer_lbs.step()

    def _forward_lbs(self):
        points = self._input_surface_points.reshape(-1, self._numPoint, 3)
        points_deformed_gt = self._deformed_points.reshape(-1, self._numPoint, 3)
        delta_beta = self._input_delta_beta.reshape(-1, self._numPoint, 3)
        smpl_tfs = self._input_smpl_tfs.reshape(-1, 24, 4, 4)
        random_points = self._input_random_points.reshape(-1, self._numPoint, 3)
        delta_beta_random = self._input_delta_beta_random.reshape(-1, self._numPoint, 3)
        
        closest_points = self._input_closest_points.reshape(-1, self._numPoint, 3)
        delta_beta_closest = self._input_delta_beta_closest.reshape(-1, self._numPoint, 3)
        closest_points_deformed_gt = self._deformed_closest_points.reshape(-1, self._numPoint, 3)

        points = closest_points
        points_deformed_gt = closest_points_deformed_gt
        delta_beta = delta_beta_closest

        poses = self._poses.reshape(-1, self.dim_theta).unsqueeze(1)
        poses = poses.repeat(1, self._numPoint, 1)

        with torch.no_grad():
            points_embed = self.embedder(points)
            random_points_embed = self.embedder(random_points)

        lbs_weight_points = self._lbs(points, None)
        lbs_weight_points_logsoft = F.log_softmax(lbs_weight_points, dim=-1)
        lbs_weight_points_softmax = lbs_weight_points.softmax(dim=-1)

        self.loss_kl = F.kl_div(lbs_weight_points_logsoft.reshape(-1, 24), self._weight_surface.reshape(-1, 24), reduction='batchmean')

        z_style = None
        
        lbs_weight_random_points = self._lbs(random_points, None)
        lbs_weight_random_points_logsoft = F.log_softmax(lbs_weight_random_points, dim=-1)
        lbs_weight_random_points_softmax = lbs_weight_random_points.softmax(dim=-1)
        self.loss_kl += F.kl_div(lbs_weight_random_points_logsoft.reshape(-1, 24), self._weight_random.reshape(-1, 24), reduction='batchmean')

        self.loss_kl /= 100

        delta_theta = self._lbs_delta(poses, points_embed, z_style)
        delta_theta_random_points = self._lbs_delta(poses, random_points_embed, z_style)

        with torch.no_grad():
            weights_points = self._blend_weight(points, None) # [B, N, V]
            weights_points = weights_points.softmax(dim=-1)
            weights_random_points = self._blend_weight(random_points, None) # [B, N, V]
            weights_random_points = weights_random_points.softmax(dim=-1)

            blend_shape = torch.einsum('bl,mkl->bmk', [self._beta, self.shapedirs])
            homogen_coord = torch.zeros(blend_shape.shape[0], blend_shape.shape[1], 1).cuda()
            blend_shape_homo = torch.cat([blend_shape, homogen_coord], dim=2)
            tfs_weighted_zero = self.tfs_weighted_zero.unsqueeze(0).repeat(self._B, 1, 1, 1)
            blend_shape_homo = torch.matmul(tfs_weighted_zero, torch.unsqueeze(blend_shape_homo, dim=-1))
            blend_shape = blend_shape_homo[:, :, :3, 0]
            delta_beta = torch.einsum('bnv,bvk->bnk', [weights_points, blend_shape])
            delta_beta_random = torch.einsum('bnv,bvk->bnk', [weights_random_points, blend_shape])

        points_deformed = self.skinning(points, delta_theta, delta_beta, lbs_weight_points_softmax, smpl_tfs, self._input_tfs_c_inv)
        self.loss_surface = (points_deformed - points_deformed_gt).pow(2).sum(-1).mean()
        points_deformed1 = self.skinning(points, delta_theta_random_points, delta_beta, lbs_weight_points_softmax, smpl_tfs, tfs_c_inv=self._input_tfs_c_inv)
        self.loss_around = (points_deformed1 - points_deformed_gt).pow(2).sum(-1).mean()

        points_plus_theta = (points + delta_theta).reshape(-1,3)
        random_points_plus_theta = (random_points + delta_theta_random_points).reshape(-1,3)

        with torch.no_grad():
            sdf_random_points = self._body_sdf(random_points.reshape(-1,3), None, len(random_points)).squeeze().reshape(self._B, -1)
            sdf_points = self._body_sdf(points.reshape(-1,3), None, len(points)).squeeze().reshape(self._B, -1)
            
        self.loss_collision_neutral, sdf_points_plus_delta = self.collision_neutral(points_plus_theta, return_sdf=True)
        loss_collision_neutral_random, self.loss_collision_order =self.collision_neutral_order(random_points_plus_theta, sdf_random_points.reshape(-1), self._order_pair.reshape(-1, 2))

        self.loss_collision_neutral /= 2
        self.loss_collision_neutral += loss_collision_neutral_random/2
        self.loss_collision_neutral *= 10
        self.loss_collision_order *= 10

        random_points_deformed = self.skinning(random_points, delta_theta_random_points, delta_beta_random, lbs_weight_random_points_softmax, smpl_tfs, tfs_c_inv=self._input_tfs_c_inv)
        self.loss_collision = self.collision_beta_theta(random_points_deformed, self._vertices_body)*100

        self.record_loss_surface.append(self.loss_surface.item())
        self.record_loss_around.append(self.loss_around.item())
        self.record_loss_collision.append(self.loss_collision.item())
        self.record_loss_collision_order.append(self.loss_collision_order.item())
        self.record_loss_collision_neutral.append(self.loss_collision_neutral.item())
        self.record_loss_kl.append(self.loss_kl.item())

        return self.loss_surface + self.loss_kl + self.loss_around + self.loss_collision + self.loss_collision_neutral + self.loss_collision_order

    def skinning(self, x, delta_theta, delta_beta, w, tfs, tfs_c_inv):
        """Linear blend skinning
        Args:
            x (tensor): canonical points. shape: [B, N, D]
            delta_theta (tensor): pose displacement. shape: [B, N, D]
            delta_beta (tensor): shape displacement. shape: [B, N, D]
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

    def collision_neutral(self, points, return_sdf=False):
        # points: (-1, 3)

        sdf_body = self._body_sdf(points, None, len(points)).squeeze()
        loss = F.relu(1e-4-sdf_body).mean()*10
        if return_sdf:
            return loss, sdf_body
        else:
            return loss

    def collision_neutral_order(self, points, points_sdf_rest, order_pair):
        # points: (-1, 3)

        vertices_body_zero = self._vertices_body_zero[0]
        faces_body = self._faces_body[0]

        sdf_body = self._body_sdf(points, None, len(points)).squeeze()
        loss = F.relu(1e-3-sdf_body).mean()*10

        with torch.no_grad():
            random_idx1 = order_pair[:, 0]
            random_idx2 = order_pair[:, 1]
            sign = torch.sign(points_sdf_rest[random_idx1] - points_sdf_rest[random_idx2])

        loss_order = F.relu(sign*(sdf_body[random_idx2]-sdf_body[random_idx1])).mean()/10

        return loss, loss_order

    def collision_beta_theta(self, points, vertices_body):
        # points: (B*numFrame, numPoint, 3)
        # vertices_body: (B, numFrame, numPoint, 3)

        epsilon = 1e-4
        faces_body = self._faces_body.cpu().numpy()[0]
        loss = 0
        for i in range(len(vertices_body)):
            body_mesh = kaolin.rep.TriangleMesh.from_tensors(vertices_body[i], self._faces_body[0])
            body_mesh.cuda()
            signed_distance_function = kaolin.conversions.trianglemesh_to_sdf(body_mesh)

            sdf_body = signed_distance_function(points[i])
            loss += F.relu(epsilon-sdf_body).mean()
        loss /= len(vertices_body)
        return loss


    def get_current_errors(self):
        loss_dict = OrderedDict([('lbs_surface', self.record_loss_surface),
                                 ('lbs_around', self.record_loss_around),
                                 ('lbs_collision', self.record_loss_collision),
                                 ('lbs_collision_order', self.record_loss_collision_order), 
                                 ('lbs_collision_neutral', self.record_loss_collision_neutral),
                                 ('lbs_kl', self.record_loss_kl),
                                 ])
        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_lbs', self._current_lr_lbs)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        if type(self.trim_mesh) == type(None):
            return visuals

        self.trim_mesh.visual.vertex_colors[:, :3] = [160, 160, 255]
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=(0, 0, 0))
        pyrender_mesh = pyrender.Mesh.from_trimesh(self.trim_mesh, smooth=False)
        
        scene.add(pyrender_mesh)

        try:
            smpl_mesh = trimesh.Trimesh(self.smpl_verts, self.smpl_faces)
            scene.add(pyrender.Mesh.from_trimesh(smpl_mesh, smooth=False))
        except:
            pass

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
        scene.add(light, pose=np.eye(4))

        dist = 3#1.5
        angle = 45
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        camera_pose = np.array([[c, 0, s, dist*s],[0, 1, 0, 0],[-1*s, 0, c, dist*c],[0, 0, 0, 1]])
        camera = pyrender.PerspectiveCamera(yfov=2/3.0, znear=0.5, zfar=5) #np.pi
        scene.add(camera, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(512, 512)
        color, depth = renderer.render(scene)

        depth[depth>0] -= 2
        depth = ((depth/np.max(depth))*255).astype(np.uint8)

        
        visuals['color_reconstruction'] = color
        visuals['depth_reconstruction'] = depth
        visuals['trimesh'] = self.trim_mesh
        self.trim_mesh = None

        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._lbs, 'lbs', label)
        self._save_network(self._lbs_delta, 'lbs_delta', label)

        # save optimizers
        self._save_optimizer(self._optimizer_lbs, 'lbs', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._lbs, 'lbs', load_epoch)
        self._load_network(self._lbs_delta, 'lbs_delta', load_epoch)

        if self._is_train and not self.use_penetration:
            # load optimizers
            self._load_optimizer(self._optimizer_lbs, 'lbs', load_epoch)

            for param_group in self._optimizer_lbs.param_groups:
                param_group['lr'] = self._current_lr_lbs
            print("Rebooting LR")
