'''
Logging and visualization logic.
'''

from __init__ import *

# Internal imports.
import logvisgen
import visualization


class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, context, log_level=None):
        if 'batch_size' in args:
            if args.is_debug:
                self.step_interval = max(16 // args.batch_size, 2)
            else:
                self.step_interval = max(64 // args.batch_size, 2)
        else:
            if args.is_debug:
                self.step_interval = 4
            else:
                self.step_interval = 16
        self.half_step_interval = self.step_interval // 2

        # With odd strides, we can interleave info from two data sources.
        if self.step_interval % 2 == 0:
            self.step_interval += 1
        if self.half_step_interval % 2 == 0:
            self.half_step_interval += 1

        # TEMP / DEBUG:
        # self.step_interval = 2

        super().__init__(log_dir=args.log_path, context=context, log_level=log_level)

    def handle_train_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                          data_retval, model_retval, loss_retval, train_args, test_args):

        if not(('train' in phase and cur_step % self.step_interval == 0) or
               ('val' in phase and cur_step % self.half_step_interval == 0) or
                ('test' in phase)):
            return

        file_idx = data_retval['file_idx'][0].item()

        # Obtain friendly short name for this step for logging, video saving, and CSV export.
        if not('test' in phase):
            file_name_suffix = ''
            file_name_suffix += f'e{epoch}_p{phase}_s{cur_step}_{file_idx}'

        else:
            file_name_suffix = ''
            file_name_suffix += f's{cur_step}_{file_idx}'

        # Log informative line including loss values & metrics in console.
        to_print = f'[Step {cur_step} / {steps_per_epoch}]  f: {file_idx}  '

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
        if log_rarely > 0 and cur_step % (self.step_interval * 16) != self.step_interval * 8:
            return file_name_suffix

        temp_st = time.time()  # DEBUG

        if model_retval is None:
            return  # Stop early if data_loop_only.

        # Save input, prediction, and ground truth data.
        rgb_input = rearrange(data_retval['rgb_input'][0],
                              'C H W -> H W C').detach().cpu().numpy()
        rgb_output = rearrange(model_retval['rgb_output'][0],
                               'C H W -> H W C').detach().cpu().numpy()
        rgb_target = rearrange(model_retval['rgb_target'][0],
                               'C H W -> H W C').detach().cpu().numpy()
        
        # self.logger.debug(
        #     f'logvis tensor to cpu: {time.time() - temp_st:.3f}s')  # DEBUG
        temp_st = time.time()  # DEBUG

        # Create simple horizontal gallery.
        vis_gal = np.concatenate([rgb_input, rgb_output, rgb_target], axis=1)
        vis_gal = np.clip(vis_gal, 0.0, 1.0)
        
        # self.logger.debug(
        #     f'logvis vis/gal creation total: {time.time() - temp_st:.3f}s')  # DEBUG
        temp_st = time.time()  # DEBUG

        if not('test' in phase):
            wandb_step = epoch
            accumulate_online = 8
        else:
            wandb_step = cur_step
            accumulate_online = 1
        
        # NOTE: Without apply_async, this part would take by far the most time.
        avoid_wandb = test_args.avoid_wandb if 'test' in phase else train_args.avoid_wandb
        online_name = f'gal_p{phase}' if avoid_wandb == 0 else None
        self.save_image(vis_gal, step=wandb_step,
                        file_name=f'{file_name_suffix}_gal.png',
                        online_name=online_name,
                        caption=file_name_suffix,
                        upscale_factor=2,
                        accumulate_online=accumulate_online,
                        apply_async=True)

        # self.logger.debug(
        #     f'logvis saving: {time.time() - temp_st:.3f}s')  # DEBUG

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

        file_name_suffix = self.handle_train_step(
            0, 'test', cur_step, cur_step, num_steps, data_retval, model_retval, loss_retval,
            all_args['train'], all_args['test'])

        return file_name_suffix
