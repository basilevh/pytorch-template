'''
Manages training & validation.
Created by Basile Van Hoorick.
'''

from __init__ import *

# Library imports.
import torch_optimizer

# Internal imports.
import args
import data
import loss
import logvis
import model
import my_utils
import pipeline


def _get_learning_rate(optimizer):
    if isinstance(optimizer, dict):
        optimizer = my_utils.any_value(optimizer)
    if optimizer is None:
        return -1.0
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _train_one_epoch(args, train_pipeline, networks_nodp, phase, epoch, optimizers,
                     lr_schedulers, data_loader, device, logger):
    assert phase in ['train', 'val', 'val_aug', 'val_noaug']

    log_str = f'Epoch (1-based): {epoch + 1} / {args.num_epochs}'
    logger.info()
    logger.info('=' * len(log_str))
    logger.info(log_str)
    if phase == 'train':
        logger.info(f'===> Train ({phase})')
        logger.report_scalar(phase + '/learn_rate', _get_learning_rate(optimizers), step=epoch)
    else:
        logger.info(f'===> Validation ({phase})')

    train_pipeline[1].set_phase(phase)
    short_desc = f'[{phase[0].upper()} {epoch + 1} / {args.num_epochs}]'

    steps_per_epoch = len(data_loader)
    total_step_base = steps_per_epoch * epoch  # This has already happened so far.
    start_time = time.time()
    num_exceptions = 0

    for cur_step, data_retval in enumerate(tqdm.rich.tqdm(data_loader, desc=short_desc)):

        if cur_step == 0:
            logger.info(f'Enter first data loader iteration took {time.time() - start_time:.3f}s')

        total_step = cur_step + total_step_base  # For continuity in wandb.
        progress = total_step / (args.num_epochs * steps_per_epoch)

        data_retval['within_batch_idx'] = torch.arange(args.batch_size)  # (B).

        def _train_step():
            # First, address every example independently.
            # This part has zero interaction between any pair of GPUs.
            if not args.data_loop_only:
                (model_retval, loss_retval) = train_pipeline[0](
                    data_retval, cur_step, total_step, epoch, progress, True, False, False)

                # Second, process accumulated information, for example contrastive loss
                # functionality. This part typically happens on the first GPU, so it should be
                # kept minimal in memory.
                loss_retval = train_pipeline[1].process_entire_batch(
                    data_retval, model_retval, loss_retval, cur_step, total_step, epoch,
                    progress)

            else:
                (model_retval, loss_retval) = (None, None)

            # Print and visualize stuff.
            logger.handle_train_step(epoch, phase, cur_step, total_step, steps_per_epoch,
                                     data_retval, model_retval, loss_retval, args, None, 0)

            # Perform backpropagation to update model parameters.
            if phase == 'train' and not args.data_loop_only:

                optimizers['backbone'].zero_grad()

                if torch.isnan(loss_retval['total']):
                    logger.warning('Skipping backbone optimizer step due to loss = NaN.')

                elif not(loss_retval['total'].requires_grad):
                    logger.warning('Skipping backbone optimizer step due to requires_grad = False.')

                else:
                    loss_retval['total'].backward()

                    # Apply gradient clipping if desired.
                    if args.gradient_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(networks_nodp['backbone'].parameters(),
                                                       args.gradient_clip)

                    optimizers['backbone'].step()

        # Don't catch exceptions when debugging.
        if args.is_debug:

            _train_step()

        else:
            try:

                _train_step()

            except Exception as e:
                num_exceptions += 1
                if num_exceptions >= 20:
                    raise e
                else:
                    logger.exception(e)
                    continue

        if cur_step >= 500 and args.is_debug:
            logger.warning('Cutting epoch short for debugging...')
            break

    if phase == 'train':
        for (k, v) in lr_schedulers.items():
            v.step()

    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    torch.cuda.empty_cache()


def _train_all_epochs(args, train_pipeline, networks_nodp, optimizers, lr_schedulers, start_epoch,
                      train_loader, val_aug_loader, val_noaug_loader, device, logger,
                      checkpoint_fn):

    logger.info(f'Start training loop of {args.name}...')
    start_time = time.time()

    for epoch in range(start_epoch, args.num_epochs):

        # Training.
        _train_one_epoch(
            args, train_pipeline, networks_nodp, 'train', epoch, optimizers,
            lr_schedulers, train_loader, device, logger)

        # Save model weights.
        checkpoint_fn(epoch)

        # Flush remaining visualizations.
        logger.epoch_finished(epoch)

        if epoch % args.val_every == 0:

            # Validation with data augmentation.
            if args.do_val_aug:
                _train_one_epoch(
                    args, train_pipeline, networks_nodp, 'val_aug', epoch, optimizers,
                    lr_schedulers, val_aug_loader, device, logger)

            # Validation without data augmentation.
            if args.do_val_noaug:
                _train_one_epoch(
                    args, train_pipeline, networks_nodp, 'val_noaug', epoch, optimizers,
                    lr_schedulers, val_noaug_loader, device, logger)

            # Flush remaining visualizations.
            if args.do_val_aug or args.do_val_noaug:
                logger.epoch_finished(epoch)  # Current impl. is fine to call more than once.

        # OPTIONAL: Keep track of best weights.

    logger.info()
    total_time = time.time() - start_time
    logger.info(f'Total time: {total_time / 3600.0:.3f} hours')


def main(args, logger):

    logger.info()
    logger.info('torch version: ' + str(torch.__version__))
    logger.info('torchvision version: ' + str(torchvision.__version__))
    logger.save_args(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    logger.info('Checkpoint path: ' + args.checkpoint_path)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    logger.info('Initializing model...')
    start_time = time.time()

    # Instantiate networks.
    backbone_args = dict()
    backbone_args['image_height'] = args.image_height
    backbone_args['image_width'] = args.image_width
    backbone_args['in_channels'] = 3
    backbone_args['out_channels'] = 3
    backbone_args['pretrained'] = True
    
    # backbone_net = model.MySimpleModel(logger, **backbone_args)
    backbone_net = model.MyDenseVitModel(logger, **backbone_args)
    
    networks = dict()
    networks['backbone'] = backbone_net

    # Bundle networks into a list.
    for (k, v) in networks.items():
        networks[k] = networks[k].to(device)
    networks_nodp = networks.copy()

    param_count = sum(p.numel() for p in backbone_net.parameters())
    logger.info(f'Backbone parameter count: {int(np.round(param_count / 1e6))}M')

    # Instantiate encompassing pipeline for more efficient parallelization.
    train_pipeline = pipeline.MyTrainPipeline(args, logger, networks, device)
    train_pipeline = train_pipeline.to(device)
    train_pipeline_nodp = train_pipeline
    if args.device == 'cuda':
        train_pipeline = torch.nn.DataParallel(train_pipeline)

    # Instantiate optimizers and learning rate schedulers.
    optimizers = dict()
    lr_schedulers = dict()
    if args.optimizer == 'sgd':
        optimizer_class = torch.optim.SGD
    elif args.optimizer == 'adam':
        optimizer_class = torch.optim.Adam
    elif args.optimizer == 'adamw':
        optimizer_class = torch.optim.AdamW
    elif args.optimizer == 'lamb':
        optimizer_class = torch_optimizer.Lamb
    milestones = [(args.num_epochs * 2) // 5,
                  (args.num_epochs * 3) // 5,
                  (args.num_epochs * 4) // 5]
    for (k, v) in networks.items():
        if len(list(v.parameters())) != 0:
            optimizers[k] = optimizer_class(v.parameters(), lr=args.learn_rate)
            lr_schedulers[k] = torch.optim.lr_scheduler.MultiStepLR(
                optimizers[k], milestones, gamma=args.lr_decay)

    # Load weights from checkpoint if specified.
    if args.resume:
        logger.info('Loading weights from: ' + args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        for (k, v) in networks_nodp.items():
            v.load_state_dict(checkpoint['net_' + k])
        for (k, v) in optimizers.items():
            v.load_state_dict(checkpoint['optim_' + k])
        for (k, v) in lr_schedulers.items():
            v.load_state_dict(checkpoint['lr_sched_' + k])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    logger.info(f'Took {time.time() - start_time:.3f}s')

    # Instantiate datasets.
    logger.info('Initializing data loaders...')
    start_time = time.time()
    (train_loader, val_aug_loader, val_noaug_loader, dset_args) = \
        data.create_train_val_data_loaders(args, logger)
    logger.info(f'Took {time.time() - start_time:.3f}s')

    # Define logic for how to store checkpoints.
    def save_model_checkpoint(epoch):
        if args.checkpoint_path:
            logger.info(f'Saving model checkpoint to {args.checkpoint_path}...')
            start_time = time.time()

            checkpoint = {
                'epoch': epoch,
                'train_args': args,
                'dset_args': dset_args,
                'backbone_args': backbone_args,
            }
            for (k, v) in networks_nodp.items():
                checkpoint['net_' + k] = v.state_dict()
            for (k, v) in optimizers.items():
                checkpoint['optim_' + k] = v.state_dict()
            for (k, v) in lr_schedulers.items():
                checkpoint['lr_sched_' + k] = v.state_dict()

            # Always update most recent checkpoint after every epoch.
            if not(args.is_debug) or epoch % args.checkpoint_every == 0 or epoch < 0:
                torch.save(checkpoint,
                           os.path.join(args.checkpoint_path, 'checkpoint.pth'))

                # Also keep track of latest epoch to allow for efficient retrieval.
                np.savetxt(os.path.join(args.checkpoint_path, 'checkpoint_epoch.txt'),
                           np.array([epoch], dtype=np.int32), fmt='%d')
                np.savetxt(os.path.join(args.checkpoint_path, 'checkpoint_name.txt'),
                           np.array([args.name]), fmt='%s')

            # Save certain fixed model epoch only once in a while.
            if epoch % args.checkpoint_every == 0 or epoch < 0:
                shutil.copy(os.path.join(args.checkpoint_path, 'checkpoint.pth'),
                            os.path.join(args.checkpoint_path, 'model_{}.pth'.format(epoch)))

            logger.info(f'Took {time.time() - start_time:.3f}s')
            logger.info()

    if args.avoid_wandb < 3:
        logger.init_wandb(PROJECT_NAME, args, networks.values(), name=args.name + '_',
                        group=args.wandb_group)

    # Print train arguments.
    logger.info('Final train command args: ' + str(args))
    logger.info('Final train dataset args: ' + str(dset_args))

    # Start training loop.
    _train_all_epochs(
        args, (train_pipeline, train_pipeline_nodp), networks_nodp, optimizers, lr_schedulers,
        start_epoch, train_loader, val_aug_loader, val_noaug_loader, device, logger,
        save_model_checkpoint)


if __name__ == '__main__':

    # DEBUG / WARNING: This is slow, but we can detect NaNs this way:
    # torch.autograd.set_detect_anomaly(True)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    args = args.train_args()

    logger = logvis.MyLogger(args, context='train', log_level=args.log_level.upper())

    if args.is_debug:

        # Don't catch exceptions when debugging.
        main(args, logger)

    else:

        try:

            main(args, logger)

        except Exception as e:

            logger.exception(e)

            logger.warning('Shutting down due to exception...')
