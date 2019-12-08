import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name, data_name, experiment_num):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}/{}'.format(model_name, data_name, experiment_num)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, d_error, e_error, epoch, n_batch, num_of_batches):

        d_error = d_error.data.cpu().numpy()
        e_error = e_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_of_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar(
            '{}/e_error'.format(self.comment), e_error, step)

    def display_status(self, epoch, num_epochs, n_batch, num_of_batches, d_error, e_error, d_pred_real, d_pred_fake):

        d_error = d_error.data.cpu().numpy()
        e_error = e_error.data.cpu().numpy()
        d_pred_real = d_pred_real.data
        d_pred_fake = d_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_of_batches)
        )
        print('Discriminator Loss: {:.4f}, Editor Loss: {:.4f}'.format(
            d_error, e_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(
            d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, editor, discriminator, epoch):
        out_dir = './model{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(editor.state_dict(),
                   '{}/E_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_of_batches):
        return epoch * num_of_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
