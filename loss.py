'''
Objective functions.
Created by Basile Van Hoorick.
'''

from __init__ import *

# Internal imports.
import metrics


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

    def per_example(self, data_retval, model_retval, progress, metrics_only):
        '''
        Loss calculations that *can* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :param metrics_only (bool).
        :return loss_retval (dict): Preliminary loss information.
        '''
        (B, H, W, _) = data_retval['rgb_input'].shape

        if metrics_only:
            # Calculate only evaluation metrics and nothing else.
            metrics_retval = metrics.calculate_metrics_for(data_retval, model_retval)
            loss_retval = dict()
            loss_retval['metrics'] = metrics_retval
            return loss_retval
        
        # Evaluate entire subbatch for efficiency.
        rgb_target = data_retval['rgb_target']
        rgb_output = model_retval['rgb_output']
        
        loss_l1 = None
        
        if self.train_args.l1_lw > 0.0:
            loss_l1 = self.my_l1_loss(rgb_output, rgb_target)
        
        # Calculate preliminary evaluation metrics.
        metrics_retval = metrics.calculate_metrics_for(data_retval, model_retval)
        
        # OPTIONAL:
        # Delete memory-taking stuff before aggregating across GPUs.
        # I observed that detaching saves around 3-4%.

        # Return results.
        loss_retval = dict()
        loss_retval['l1'] = loss_l1
        loss_retval['metrics'] = metrics_retval
        return loss_retval

    def entire_batch(self, data_retval, model_retval, loss_retval, cur_step, total_step, epoch,
                     progress):
        '''
        Loss calculations that *cannot* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :return loss_retval (dict): All loss information.
        '''
        # For debugging:
        old_loss_retval = loss_retval.copy()
        old_loss_retval['metrics'] = loss_retval['metrics'].copy()

        if not('test' in self.phase):
            # Log average value per epoch at train / val time.
            key_prefix = self.phase + '/'
            report_kwargs = dict(remember=True)
        else:
            # Log & plot every single step at test time.
            key_prefix = ''
            report_kwargs = dict(step=cur_step)

        if len(loss_retval.keys()) >= 2:  # Otherwise, assume we had metrics_only enabled.

            # Average all loss values across batch size.
            for (k, v) in loss_retval.items():
                if not('metrics' in k):
                    if torch.is_tensor(v):
                        loss_retval[k] = torch.mean(v)
                    elif v is None:
                        loss_retval[k] = -1.0
                    else:
                        raise RuntimeError(f'loss_retval: {k}: {v}')

            # Obtain total loss per network.
            loss_total = loss_retval['l1'] * self.train_args.l1_lw

            # Convert loss terms (just not the total) to floats for logging.
            for (k, v) in loss_retval.items():
                if torch.is_tensor(v):
                    loss_retval[k] = v.item()

            # Report all loss values.
            self.logger.report_scalar(
                key_prefix + 'loss_total', loss_total.item(), **report_kwargs)
            if self.train_args.l1_lw > 0.0:
                self.logger.report_scalar(
                    key_prefix + 'loss_l1', loss_retval['l1'], **report_kwargs)

            # Return results, i.e. append new stuff to the existing loss_retval dictionary.
            # Total losses are the only entries that are tensors, not just floats.
            # Later in train.py, we will match the appropriate optimizer (and thus network parameter
            # updates) to each accumulated loss value as indicated by the keys here.
            loss_retval['total'] = loss_total

        # Weighted average all metrics across batch size.
        # TODO DRY: This is also in metrics.py.
        for (k, v) in loss_retval['metrics'].items():
            if 'count' in k:
                count_key = k
                mean_key = k.replace('count', 'mean')
                short_key = k.replace('count_', '')
                old_counts = loss_retval['metrics'][count_key]
                old_means = loss_retval['metrics'][mean_key]

                # NOTE: Some mean values will be -1.0 but then corresponding counts are always 0.
                new_count = old_counts.sum().item()
                if new_count > 0:
                    new_mean = (old_means * old_counts).sum().item() / (new_count + 1e-7)
                else:
                    new_mean = -1.0
                loss_retval['metrics'][count_key] = new_count
                loss_retval['metrics'][mean_key] = new_mean

                # Report all metrics, but ignore invalid values (e.g. when no occluded or contained
                # stuff exists). At train time, we maintain correct proportions with the weight
                # option. At test time, we log every step anyway, so this does not matter.
                if new_count > 0:
                    self.logger.report_scalar(key_prefix + short_key, new_mean, weight=new_count,
                                              **report_kwargs)

        return loss_retval
