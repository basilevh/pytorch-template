'''
Objective functions.
'''

from __init__ import *


class MyLosses():
    '''
    Wrapper around the loss functionality such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, phase):
        self.train_args = train_args
        self.logger = logger
        self.phase = phase
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def my_l1_loss(self, rgb_output, rgb_target):
        '''
        :param rgb_output (B, H, W, 3) tensor.
        :param rgb_target (B, H, W, 3) tensor.
        :return loss_l1 (tensor).
        '''
        loss_l1 = self.l1_loss(rgb_output, rgb_target)
        return loss_l1

    def per_example(self, data_retval, model_retval):
        '''
        Loss calculations that *can* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :return loss_retval (dict): Preliminary loss information.
        '''
        (B, H, W, _) = data_retval['rgb_input'].shape

        loss_l1 = []

        # OPTIONAL:
        # Loop over every example.
        # for i in range(B):

        #     rgb_input = data_retval['rgb_input'][i:i + 1]
        #     rgb_output = model_retval['rgb_output'][i:i + 1]
        #     rgb_target = data_retval['rgb_target'][i:i + 1]

        #     # Calculate loss terms.
        #     cur_l1 = self.my_l1_loss(rgb_output, rgb_target)
            
        #     # Update lists.
        #     if self.train_args.l1_lw > 0.0:
        #         loss_l1.append(cur_l1)

        # # Average & return losses + other informative metrics across batch size within this GPU.
        # loss_l1 = torch.mean(torch.stack(loss_l1)) if self.train_args.l1_lw > 0.0 else None

        # PREFERRED:
        # Evaluate entire subbatch for efficiency.
        rgb_output = model_retval['rgb_output']
        rgb_target = data_retval['rgb_target']
        
        # Update loss terms.
        if self.train_args.l1_lw > 0.0:
            loss_l1 = self.my_l1_loss(rgb_output, rgb_target)

        else:
            loss_l1 = None
        
        # OPTIONAL:
        # Delete memory-taking stuff before aggregating across GPUs.
        # I observed that detaching saves around 3-4%.

        # Return results.
        loss_retval = dict()
        loss_retval['l1'] = loss_l1
        return loss_retval

    def entire_batch(self, data_retval, model_retval, loss_retval, epoch_frac):
        '''
        Loss calculations that *cannot* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param epoch_frac (float): Current epoch (0-based) divided by total number of epochs.
        :return loss_retval (dict): All loss information.
        '''

        # Average all terms across batch size.
        for (k, v) in loss_retval.items():
            if torch.is_tensor(v):
                loss_retval[k] = torch.mean(v)
            elif v is None:
                loss_retval[k] = 0.0

        # Obtain total loss. 
        loss_total = loss_retval['l1'] * self.train_args.l1_lw
        
        # Convert loss terms (just not the total) to floats for logging.
        for (k, v) in loss_retval.items():
            if torch.is_tensor(v):
                loss_retval[k] = v.item()

        # Report all loss values.
        self.logger.report_scalar(
            self.phase + '/loss_total', loss_total.item(), remember=True)
        self.logger.report_scalar(
            self.phase + '/loss_l1', loss_retval['l1'], remember=True)

        # Return results, i.e. append to the existing loss_retval dictionary.
        # Total losses are the only entries that are tensors, not just floats.
        loss_retval['total'] = loss_total
        return loss_retval
