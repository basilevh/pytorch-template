'''
Logging and visualization logic.
'''

from __init__ import *

# Internal imports.
import logvisgen


class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, context):
        if 'batch_size' in args:
            if args.is_debug:
                self.step_interval = 80 // args.batch_size
            else:
                self.step_interval = 3200 // args.batch_size
        else:
            if args.is_debug:
                self.step_interval = 20
            else:
                self.step_interval = 200
        super().__init__(args.log_path, context)

    def handle_train_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                          data_retval, model_retval, loss_retval, train_args):

        color_map = plt.get_cmap('magma')

        if cur_step % self.step_interval == 0:

            batch_size = data_retval['rgb_input'].shape[0]
            total_loss = loss_retval['total']
            loss_l1 = loss_retval['l1']

            # Print metrics in console.
            self.info(f'[Step {cur_step} / {steps_per_epoch}]  '
                      f'total_loss: {total_loss:.3f}  '
                      f'loss_l1: {loss_l1:.3f}')

            # Save metadata in JSON format, for eventual inspection / debugging / reproducibility.
            metadata = dict()
            metadata['batch_size'] = batch_size

            # Save input, prediction, and ground truth images.
            rgb_input = data_retval['rgb_input'][0].permute(1, 2, 0).detach().cpu().numpy()
            rgb_output = model_retval['rgb_output'][0].permute(1, 2, 0).detach().cpu().numpy()
            rgb_target = 1.0 - rgb_input

            gallery = np.stack([rgb_input, rgb_output, rgb_target])
            gallery = np.clip(gallery, 0.0, 1.0)
            self.save_gallery(gallery, step=epoch,
                              file_name=f'rgb_e{epoch}_p{phase}_s{cur_step}.png',
                              online_name=f'rgb_p{phase}')
            
            self.save_text(metadata, file_name=f'metadata_e{epoch}_p{phase}_s{cur_step}.txt')

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)
        self.commit_scalars(step=epoch)

    def handle_test_step(self, cur_step, num_steps, data_retval, inference_retval, all_args):
        '''
        :param all_args (dict): train, test, train_dset, test_dset, model.
        '''

        color_map = plt.get_cmap('magma')
        batch_size = data_retval['rgb_input'].shape[0]
        psnr = inference_retval['psnr']

        # Print metrics in console.
        self.info(f'[Step {cur_step} / {num_steps}]  '
                  f'psnr: {psnr.mean():.2f} ± {psnr.std():.2f}')
            
        # Save metadata in JSON format, for inspection / debugging / reproducibility.
        metadata = dict()
        metadata['batch_size'] = batch_size

        # Save input, prediction, and ground truth images.
        rgb_input = inference_retval['rgb_input']
        rgb_output = inference_retval['rgb_output']
        rgb_target = inference_retval['rgb_target']

        gallery = np.stack([rgb_input, rgb_output, rgb_target])
        gallery = np.clip(gallery, 0.0, 1.0)
        self.save_gallery(gallery, step=cur_step,
                          file_name=f'rgb_iogt_s{cur_step}.png',
                          online_name=f'rgb_iogt')

        self.save_text(metadata, file_name=f'metadata_s{cur_step}.txt')

