'''
Entire training pipeline logic.
'''

from __init__ import *


# Internal imports.
import loss
import my_utils


class MyTrainPipeline(torch.nn.Module):
    '''
    Wrapper around most of the training iteration such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, networks, device):
        super().__init__()
        self.train_args = train_args
        self.logger = logger
        self.networks = torch.nn.ModuleDict(networks)
        self.device = device
        self.phase = None  # Assigned only by set_phase().
        self.losses = None  # Instantiated only by set_phase().

    def set_phase(self, phase):
        '''
        Must be called when switching between train / validation / test phases.
        '''
        self.phase = phase
        self.losses = loss.MyLosses(self.train_args, self.logger, phase)

        if 'train' in phase:
            self.train()
            for (k, v) in self.networks.items():
                v.train()
            torch.set_grad_enabled(True)

        else:
            self.eval()
            for (k, v) in self.networks.items():
                v.eval()
            torch.set_grad_enabled(False)

    def forward(self, data_retval, cur_step, total_step, epoch, progress, include_loss,
                metrics_only, no_pred):
        '''
        Handles one parallel iteration of the training or validation phase.
        Executes the models and calculates the per-example losses.
        This is all done in a parallelized manner to minimize unnecessary communication.
        :param data_retval (dict): Data loader elements.
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :param epoch (int): Current epoch index (0-based).
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :param include_loss (bool).
        :param metrics_only (bool).
        :param no_pred (bool).
        :return (model_retval, loss_retval).
            model_retval (dict): All output information.
            loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        '''

        within_batch_inds = data_retval['within_batch_idx']
        B = within_batch_inds.shape[0]

        # Retrieve data.
        rgb_input = data_retval['rgb_input']
        rgb_input = rgb_input.to(self.device)

        # Run model.
        rgb_output = self.networks['backbone'](rgb_input)

        # Organize and return relevant info.
        model_retval = dict()
        model_retval['rgb_target'] = 1.0 - rgb_input
        model_retval['rgb_output'] = rgb_output

        if include_loss:
            loss_retval = self.losses.per_example(data_retval, model_retval, progress, metrics_only)
        else:
            loss_retval = None

        return (model_retval, loss_retval)

    def process_entire_batch(self, data_retval, model_retval, loss_retval, cur_step, total_step,
                             epoch, progress):
        '''
        Finalizes the training step. Calculates all losses.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :param epoch (int): Current epoch index (0-based).
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :return loss_retval (dict): All loss information.
        '''
        loss_retval = self.losses.entire_batch(
            data_retval, model_retval, loss_retval, cur_step, total_step, epoch, progress)

        return loss_retval
