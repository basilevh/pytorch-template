'''
Evaluation logic.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'eval/'))

from __init__ import *

# Internal imports.
import args
import data
import inference
import loss
import logvis
import my_utils


def _test(all_args, networks, data_loader, device, logger):
    '''
    :param all_args (dict): train, test, train_dset, test_dset, model.
    '''
    for net in networks.values():
        net.eval()
    torch.set_grad_enabled(False)

    num_steps = len(data_loader)

    start_time = time.time()

    for cur_step, data_retval in enumerate(tqdm.tqdm(data_loader)):

        if cur_step == 0:
            logger.info(f'Enter first data loader iteration took {time.time() - start_time:.3f}s')

        # Perform inference (independently per example).
        inference_retval = inference.perform_inference(
            data_retval, networks, device, logger, all_args)

        # Print and visualize stuff.
        logger.handle_test_step(cur_step, num_steps, data_retval, inference_retval, all_args)


def main(test_args, logger):

    logger.info()
    logger.info('torch version: ' + str(torch.__version__))
    logger.info('torchvision version: ' + str(torchvision.__version__))
    logger.save_args(test_args)

    np.random.seed(test_args.seed)
    random.seed(test_args.seed)
    torch.manual_seed(test_args.seed)
    if test_args.device == 'cuda':
        torch.cuda.manual_seed_all(test_args.seed)

    logger.info('Initializing model...')
    start_time = time.time()

    # Instantiate networks and load weights.
    if test_args.device == 'cuda':
        device = torch.device('cuda:' + str(test_args.gpu_id))
    else:
        device = torch.device(test_args.device)
    (networks, train_args, train_dset_args, model_args, epoch) = \
        inference.load_networks(test_args.resume, device, logger, epoch=test_args.epoch)
    test_args.name += f'_e{epoch}'

    logger.info(f'Took {time.time() - start_time:.3f}s')
    logger.info('Initializing data loader...')
    start_time = time.time()

    # Instantiate dataset.
    (test_loader, test_dset_args) = data.create_test_data_loader(
        train_args, test_args, train_dset_args, logger)

    logger.info(f'Took {time.time() - start_time:.3f}s')

    logger.init_wandb(PROJECT_NAME + '_test', test_args, networks.values(), name=test_args.name,
                        group='test_debug' if test_args.is_debug else 'test')

    # Print test arguments.
    logger.info('Train command args: ' + str(train_args))
    logger.info('Train dataset args: ' + str(train_dset_args))
    logger.info('Final test command args: ' + str(test_args))
    logger.info('Final test dataset args: ' + str(test_dset_args))
    
    # Combine arguments for later use.
    all_args = dict()
    all_args['train'] = train_args
    all_args['test'] = test_args
    all_args['train_dset'] = train_dset_args
    all_args['test_dset'] = test_dset_args
    all_args['model'] = model_args

    # Run actual test loop.
    _test(all_args, networks, test_loader, device, logger)


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    test_args = args.test_args()

    logger = logvis.MyLogger(test_args, context='test_' + test_args.name)

    if test_args.is_debug:

        # Don't catch exceptions when debugging.
        main(test_args, logger)

    else:

    try:

        main(test_args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')
