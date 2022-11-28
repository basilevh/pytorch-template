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
            elif args.single_scene:
                self.step_interval = max(32 // args.batch_size, 2)
            else:
                self.step_interval = max(64 // args.batch_size, 2)
        else:
            if args.is_debug:
                self.step_interval = 4
            elif args.single_scene:
                self.step_interval = 8
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

        file_name_suffix = ''

        # TODO async etc

        total_loss = loss_retval['total']
        loss_l1 = loss_retval['l1']

        # Print metrics in console.
        self.info(f'[Step {cur_step} / {steps_per_epoch}]  '
                  f'total_loss: {total_loss:.3f}  '
                  f'loss_l1: {loss_l1:.3f}')

        # Save input, prediction, and ground truth images.
        rgb_input = data_retval['rgb_input'][0].permute(1, 2, 0).detach().cpu().numpy()
        rgb_output = model_retval['rgb_output'][0].permute(1, 2, 0).detach().cpu().numpy()
        rgb_target = 1.0 - rgb_input

        temp_st = time.time()  # DEBUG
        
        gallery = np.stack([rgb_input, rgb_output, rgb_target])
        gallery = np.clip(gallery, 0.0, 1.0)
        self.save_gallery(gallery, step=epoch,
                          file_name=f'rgb_e{epoch}_p{phase}_s{cur_step}.png',
                          online_name=f'rgb_p{phase}')

        self.logger.debug(
            f'logvis saving: {time.time() - temp_st:.3f}s')  # DEBUG

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
