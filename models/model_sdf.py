import os, sys
import numpy as np
import trimesh
import pyrender
import torch
import torch.nn.functional as F
from collections import OrderedDict
from .models import BaseModel
from networks import IGR, learnt_representations
from utils.sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self._name = 'IGR_sdf'

        # create network
        self.parallel = opt.parallel
        self.position_in = 3 
        self.skip_in = [4]
        self.numG = opt.numG
        self.d_in = self.position_in + 12
        self._init_create_networks()

        self.is_trousers = opt.is_trousers

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

        if self._is_train:
            self.grad_lambda = 0.02
            self.normals_lambda = 1
            self.off_surface_lambda = 0.1
            self.sign_lambda = 1
  
            self.lambda_reg_rep = 1e-3


    def _init_create_networks(self):
        # generator network
        self._G = IGR.ImplicitNet_multiG(d_in=self.d_in, skip_in=self.skip_in)
        print(self._G)
        if torch.cuda.is_available():
            self._G.cuda()

        self._rep = learnt_representations.Network(cloth_rep_size=12, samples=self.numG)
        self._rep.init_weights()
        print(self._rep)
        if torch.cuda.is_available():
            self._rep.cuda()

    def _init_parallel(self):
        self._G = torch.nn.DataParallel(self._G)
        self._rep = torch.nn.DataParallel(self._rep)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G)
        self._optimizer_rep = torch.optim.Adam(self._rep.parameters(), lr=self._current_lr_G)

    def set_epoch(self, epoch):
        self.iepoch = epoch

    def _init_losses(self):

        self.record_loss_surface = []
        self.record_loss_sign = []
        self.record_loss_grad = []
        self.record_loss_off_surface_reg = []

        self.record_loss_reg_rep = []
        self.record_max_weight_rep = []

        self.record_loss_sdf = []

    def set_input(self, input):

        self._input_index_cloth = input['index_cloth']
        self._input_random_points = input['random_points'].float()
        self._vertices_body = input['vertices_body']
        self._faces_body = input['faces_body']
        self._beta = input['beta']
        self._gt_sdf_random_points = input['sdf']
        self._flag_surface = input['flag_surface']
        self._surface_normals = input['surface_normals']
        
        if torch.cuda.is_available():
            self._input_index_cloth = self._input_index_cloth.cuda()
            self._input_random_points = self._input_random_points.cuda()
            self._vertices_body = self._vertices_body.cuda()
            self._faces_body = self._faces_body.cuda()
            self._beta = self._beta.cuda()
            self._gt_sdf_random_points = self._gt_sdf_random_points.cuda()
            self._flag_surface = self._flag_surface.cuda()
            self._flag_not_surface = torch.logical_not(self._flag_surface)
            self._surface_normals = self._surface_normals.cuda()

        self._B = self._input_index_cloth.size(0)

        return 

    def set_train(self):
        self._G.train()
        self._rep.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._rep.eval()
        self._is_train = False

    def reconstruct(self, smpl_points=None, resolution=256, thresh = 0, just_vf=False):
        #b_min = np.array([-0.85, -0.5, -0.3])
        #b_max = np.array([0.85, 0.6, 0.3])
        
        #pants_b_min = [-0.3, -1.2, -0.3]
        #pants_b_max = [0.3, 0.0, 0.3]
        if self.is_trousers:
            b_min = np.array([-0.4, -0.6, -0.95])
            b_max = np.array([0.4, 0.6, 0.4])
        else:
            b_min = np.array([-0.8, -0.6, -0.5])
            b_max = np.array([0.8, 0.6, 0.8])

        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)

        def eval_func_xyz(points):
            points = torch.FloatTensor(points).transpose(1, 0).cuda()
            num_points = len(points)
            encode_points = points

            #print(points.shape)
            x_cloth_points = self.cloth_representation[[0]].unsqueeze(1).repeat(1, num_points, 1)
            x_cloth_points = x_cloth_points.reshape(-1, x_cloth_points.shape[-1])
            if self.numG == 1:
                x_cloth_points = torch.zeros_like(x_cloth_points).cuda()

            pred = self._G(encode_points, x_cloth_points, num_points)
            pred *= -100
            pred = pred.reshape(1,-1)
            return pred.cpu().data.numpy()

        sdf = eval_grid_octree(coords, eval_func_xyz, threshold = 0.01, num_samples=10000, init_resolution=16)

        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thresh)

        cloth_mesh = trimesh.Trimesh(np.float64(verts), faces[:, ::-1])
        cloth_mesh.vertices /= resolution
        cloth_mesh.vertices *= (b_max - b_min)[None]
        cloth_mesh.vertices += b_min[None]

        if just_vf:
            return cloth_mesh.vertices, cloth_mesh.faces 

        self.trim_mesh = cloth_mesh
        return self.trim_mesh

    def optimize_parameters(self):
        if self._is_train:
            # convert tensor to variables
            self._optimizer_rep.zero_grad()
            self._optimizer_G.zero_grad()
            loss_G = self._forward_G()
            loss_G.backward()
            
            self._optimizer_G.step()
            self._optimizer_rep.step()

    def _forward_G(self):

        gt_sdf = self._gt_sdf_random_points.reshape(-1)
        points_input = self._input_random_points
        num_points = points_input.shape[1]
        points_input = points_input.reshape(-1, self.position_in)
        points_input.requires_grad_()

        flag_surface = self._flag_surface.reshape(-1)
        flag_not_surface = self._flag_not_surface.reshape(-1)

        surface_normals = self._surface_normals.reshape(-1, 3)

        cloth_representation = self._rep(self._input_index_cloth)
        self.cloth_representation = cloth_representation.clone().detach()

        x_cloth_points = cloth_representation.unsqueeze(1).repeat(1, num_points, 1)
        x_cloth_points = x_cloth_points.reshape(-1, x_cloth_points.shape[-1])

        sdf_pred = self._G(points_input, x_cloth_points, num_points)
        
        points_grad = IGR.gradient(points_input, sdf_pred)
        self.loss_grad  = (points_grad[flag_surface] - surface_normals[flag_surface]).norm(2, dim=1).mean() + ((points_grad.norm(2, dim=-1) - 1) ** 2).mean()*0.1

        sdf_pred = sdf_pred.squeeze()
        
        self.loss_surface = sdf_pred[flag_surface].abs().mean() + (sdf_pred[flag_not_surface] - gt_sdf[flag_not_surface]).abs().mean()
        self.loss_reg_rep = cloth_representation.norm(dim=1).mean()*self.lambda_reg_rep

        self.record_loss_surface.append(self.loss_surface.item())
        self.record_loss_reg_rep.append(self.loss_reg_rep.item())
        if self.parallel:
            self.record_max_weight_rep.append(self._rep.module.weights.max().item())
        else:
            self.record_max_weight_rep.append(self._rep.weights.max().item())
        self.record_loss_grad.append(self.loss_grad.item())

        return self.loss_surface + self.loss_reg_rep + self.loss_grad

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_surface', self.record_loss_surface),
                                 ('reg_rep', self.record_loss_reg_rep),
                                 ('max_weight_rep', self.record_max_weight_rep),
                                 ('loss_grad', self.record_loss_grad),
                                 ])
        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G)])

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
        self._save_network(self._G, 'G', label)
        self._save_network(self._rep, 'rep', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_rep, 'rep', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)
        self._load_network(self._rep, 'rep', load_epoch)

        if self._is_train:
            # load optimizers
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)
            self._load_optimizer(self._optimizer_rep, 'rep', load_epoch)
            print('reset self._current_lr_G:', self._current_lr_G)
            for param_group in self._optimizer_G.param_groups:
                param_group['lr'] = self._current_lr_G
            for param_group in self._optimizer_rep.param_groups:
                if self.fast_rep:
                    param_group['lr'] = self._current_lr_G*2
                else:
                    param_group['lr'] = self._current_lr_G
            print("Rebooting LR")
