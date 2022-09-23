import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--dataset_dir', type=str, default='/cvlabsrc1/home/ren/CLOTH3D/toy_train_garment')
        self._parser.add_argument('--data_folder', type=str, default='mesh-trimesh-single-side')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--batch_size', type=int, default=6, help='input batch size')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--dataset_mode', type=str, default='bcnet_tshirts_simple', help='chooses dataset to be used')
        self._parser.add_argument('--model', type=str, default='model_smplicit', help='')
        self._parser.add_argument('--n_threads_test', default=0, type=int, help='# threads for loading data')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self._parser.add_argument('--eps', default=0.01, type=float, help='thickness of the cloth')

        self._parser.add_argument('--numG', default=1, type=int, help='number of garments')
        self._parser.add_argument('--parallel', action='store_true', help='use multiple GPU to train')

        self._parser.add_argument('--deshape', action='store_true', help='only deshaped data for training')

        self._parser.add_argument('--verts_weight', action='store_true', help='predict weights for diffusion shape')

        self._parser.add_argument('--is_trousers', action='store_true', help='if the garment is trousers')


        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        #print(models_dir)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            print('self._opt.load_epoch < 1 ',self._opt.load_epoch < 1, self._opt.load_epoch)
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
