'''
Logging and visualization logic.
Created by Basile Van Hoorick.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'utils/'))
sys.path.insert(0, os.getcwd())

from __init__ import *

# Internal imports.
import logvisgen
import visualization


class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, context, log_level=None):
        if args.is_debug:
            self.step_interval = max(64 // args.batch_size, 2)
        else:
            self.step_interval = max(256 // args.batch_size, 2)
        self.half_step_interval = self.step_interval // 2

        # With odd strides, we can interleave info from two data sources.
        if self.step_interval % 2 == 0:
            self.step_interval += 1
        if self.half_step_interval % 2 == 0:
            self.half_step_interval += 1

        super().__init__(log_dir=args.log_path, context=context, log_level=log_level)

    def handle_train_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                          data_retval, model_retval, loss_retval, train_args, test_args,
                          batch_idx):
        '''
        :param epoch (int).
        :param phase (str).
        :param cur_step (int): Which batch within current epoch.
        :param total_step (int): Which batch overall.
        :param steps_per_epoch (int): Number of batches per epoch.
        :param data_retval (dict).
        :param model_retval (dict).
        :param loss_retval (dict).
        :param train_args (dict).
        :param test_args (dict).
        :param batch_idx (int): Index within batch we wish to visualize here.
        :return file_name_suffix (str): Identifying suffix to append to file names for saving.
        '''

        if 'train' in phase:
            used_step_interval = self.step_interval
        elif 'val' in phase:
            used_step_interval = self.half_step_interval
        elif 'test' in phase:
            used_step_interval = 1
        else:
            raise ValueError(f'Unknown phase: {phase}')
        if cur_step % used_step_interval != 0:
            return
        is_sixteenth_step = (cur_step % (used_step_interval * 16) == used_step_interval * 8)

        B = data_retval['dset_idx'].shape[0]
        file_idx = data_retval['file_idx'][batch_idx].item()

        # Obtain friendly short name for this step for logging, video saving, and CSV export.
        if 'train' in phase or 'val' in phase:
            # NOTE: Typically, batch_idx == 0 in this case (or at least, we only bother with showing
            # at most one example per batch / training iteration).
            file_name_suffix = ''
            file_name_suffix += f'e{epoch}_p{phase}_s{cur_step}_f{file_idx}'

        elif 'test' in phase:
            # NOTE: If batch_idx > 0, then cur_step will not be unique per example.
            file_name_suffix = ''
            file_name_suffix += f's{cur_step}_b{batch_idx}_f{file_idx}'

        # Log informative line including loss values & metrics in console.
        if 'train' in phase or 'val' in phase:
            to_print = (f'[Step {cur_step} / {steps_per_epoch}]  '
                        f'f: {file_idx}  ')
        elif 'test' in phase:
            to_print = (f'[Step {cur_step} / {steps_per_epoch}]  '
                        f'b: {batch_idx}  f: {file_idx}  ')

        # NOTE: All wandb stuff for reporting scalars is handled in loss.py.
        # Assume loss may be missing (e.g. at test time).
        if loss_retval is not None:

            if len(loss_retval.keys()) >= 2:
                total_loss = loss_retval['total'].item()
                loss_l1 = loss_retval['l1']
                to_print += (f'tot: {total_loss:.3f}  '
                             f'l1: {loss_l1:.3f}  ')

            # Assume metrics are always present (even if count = 0).
            metrics_retval = loss_retval['metrics']

        self.info(to_print)

        # If log_rarely is active, then write stuff to disk much less often, after printing info.
        log_rarely = (0 if 'test' in phase else train_args.log_rarely)
        if log_rarely > 0 and not(is_sixteenth_step):
            return file_name_suffix

        if model_retval is None:
            return  # Stop early if data_loop_only.

        # Save input, prediction, and ground truth data.
        rgb_input = rearrange(data_retval['rgb_input'][batch_idx],
                              'C H W -> H W C').detach().cpu().numpy()
        rgb_output = rearrange(model_retval['rgb_output'][batch_idx],
                               'C H W -> H W C').detach().cpu().numpy()
        rgb_target = rearrange(model_retval['rgb_target'][batch_idx],
                               'C H W -> H W C').detach().cpu().numpy()
        
        # Create simple horizontal gallery.
        vis_gal = np.concatenate([rgb_input, rgb_output, rgb_target], axis=1)
        vis_gal = np.clip(vis_gal, 0.0, 1.0)
        
        if 'train' in phase or 'val' in phase:
            wandb_step = epoch
            accumulate_online = 8
        elif 'test' in phase:
            wandb_step = cur_step
            accumulate_online = 1

        # NOTE: Without apply_async, this part would take by far the most time.
        avoid_wandb = test_args.avoid_wandb if 'test' in phase else train_args.avoid_wandb
        if avoid_wandb == 0 or (avoid_wandb <= 1 and is_sixteenth_step):
            online_name = f'gal_p{phase}'
        else:
            online_name = None
        self.save_image(vis_gal, step=wandb_step,
                        file_name=f'gal_{file_name_suffix}.png',
                        online_name=online_name,
                        caption=file_name_suffix,
                        upscale_factor=2,
                        accumulate_online=accumulate_online,
                        apply_async=True)

        return file_name_suffix

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)
        self.commit_scalars(step=epoch)

    def handle_test_step(self, cur_step, num_steps, data_retval, inference_retval, all_args):
        '''
        :param all_args (dict): train, test, train_dset, test_dset, model.
        '''
        model_retval = inference_retval['model_retval']
        loss_retval = inference_retval['loss_retval']
        B = data_retval['dset_idx'].shape[0]

        for b in range(B):
            file_name_suffix = self.handle_train_step(
                0, 'test', cur_step, cur_step, num_steps, data_retval, model_retval, loss_retval,
                all_args['train'], all_args['test'], b)

        return file_name_suffix
