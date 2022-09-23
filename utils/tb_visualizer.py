import numpy as np
import os
import time
from . import util
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


class TBVisualizer:
    def __init__(self, opt):
        self._opt = opt
        self._save_path = os.path.join(opt.checkpoints_dir, opt.name)

        self._log_path = os.path.join(self._save_path, 'loss_log2.txt')
        self._tb_path = os.path.join(self._save_path, 'summary.json')
        self._writer = SummaryWriter(self._save_path)

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self._writer.close()

    def display_current_results(self, visuals, it, is_train, save_visuals=False):
        for label, image_numpy in visuals.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            if save_visuals:
                if label == 'trimesh':
                    save_path = os.path.join(self._opt.checkpoints_dir, self._opt.name, 'event_imgs', sum_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    image_numpy.export(os.path.join(save_path, '%08d.obj' % it), include_color=True)
                else:
                    util.save_image(image_numpy, os.path.join(self._opt.checkpoints_dir, self._opt.name,
                                                    'event_imgs', sum_name, '%08d.png' % it))

                

        self._writer.export_scalars_to_json(self._tb_path)

    def plot_scalars(self, scalars, it, is_train):
        for label, scalar in scalars.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            self._writer.add_scalar(sum_name, scalar, it)

    def plot_losses(self, losses, it, is_train):
        for label, loss in losses.items():
            if len(loss)==0:
                continue

            x = list(range(len(loss)))
            y = loss
            plt.plot(x, y, 'b')
            plt.axis([0, len(loss), 0, max(loss[10:])])
            plt.ylabel(label + ' Loss')
            plt.xlabel('Iter')
            plt.title(label)
            sum_name = '{}/Loss'.format('Train' if is_train else 'Test')
            save_path = os.path.join(self._opt.checkpoints_dir, self._opt.name, 'event_imgs', sum_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, '%s.png' % label))
            plt.clf()

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = '%s (T%s, epoch: %d, it: %d/%d, t/smpl: %.5fs) ' % (log_time, visuals_info, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            if len(v)==0:
                continue
            message += '%s:%.5f ' % (k, v[-1])

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_validate_errors(self, epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds) ' % (log_time, epoch, t)
        for k, v in errors.items():
            message += '%s:%.5f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, visuals):
        for label, image_numpy in visuals.items():
            image_name = '%s.png' % label
            save_path = os.path.join(self._save_path, "samples", image_name)
            util.save_image(image_numpy, save_path)
