'''
Handling of parameters that can be passed to training and testing scripts.
Created by Basile Van Hoorick.
'''

from __init__ import *


# Internal imports.
import my_utils


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _arg2str(arg_value):
    if isinstance(arg_value, bool):
        return '1' if arg_value else '0'
    else:
        return str(arg_value)


def shared_args(parser):
    '''
    These parameters can be passed to both training and testing / evaluation files.
    '''

    # Misc options.
    parser.add_argument('--seed', default=2025, type=int,
                        help='Random number generator seed.')
    parser.add_argument('--log_level', default='info', type=str,
                        choices=['debug', 'info', 'warn'],
                        help='Threshold for command line output.')

    # Resource options.
    parser.add_argument('--device', default='cuda', type=str,
                        choices=['cuda', 'cpu'],
                        help='cuda or cpu.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size during training or testing.')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of data loading workers; -1 means automatic.')

    # Logging & checkpointing options.
    parser.add_argument('--checkpoint_root', default='checkpoints/', type=str,
                        help='Path to parent collection of checkpoint folders.')
    parser.add_argument('--log_root', default='logs/', type=str,
                        help='Path to parent collection of logs, visualizations, and results.')
    parser.add_argument('--name', '--tag', default='', type=str,
                        help='Recognizable, unique tag of this experiment for bookkeeping. A good '
                        'practice would be to include a version number.')
    parser.add_argument('--resume', '--checkpoint_name', default='', type=str,
                        help='Tag of checkpoint to resume from. This has to match an experiment '
                        'name that is available under checkpoint_root.')
    parser.add_argument('--epoch', default=-1, type=int,
                        help='If >= 0, desired model epoch to evaluate or resume from (0-based), '
                        'otherwise pick latest.')
    parser.add_argument('--avoid_wandb', default=1, type=int,
                        help='If 1, rarely log visuals online. '
                        'If 2, do not log visuals online. '
                        'If 3, do not log anything online.')
    parser.add_argument('--log_rarely', default=0, type=int,
                        help='If 1, rarely create and store visuals.')
                        
    # Data options (all phases).
    parser.add_argument('--data_path', default=[''], type=str, nargs='+',
                        help='Path to dataset root folder(s).')
    parser.add_argument('--fake_data', default=False, type=_str2bool,
                        help='To quickly test GPU memory (VRAM) usage.')
    parser.add_argument('--use_data_frac', default=1.0, type=float,
                        help='If < 1.0, use a smaller dataset.')
    parser.add_argument('--data_loop_only', default=False, type=_str2bool,
                        help='Loop over data loaders only without any neural network operation. '
                        'This is useful for debugging and visualizing data / statistics.')

    # Automatically inferred options (do not assign).
    parser.add_argument('--is_debug', default=False, type=_str2bool,
                        help='Shorter epochs; log and visualize more often.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='Path to current checkpoint directory for this experiment.')
    parser.add_argument('--train_log_path', default='', type=str,
                        help='Path to current train-time logging directory for this experiment.')
    parser.add_argument('--log_path', default='', type=str,
                        help='Switches to train or test depending on the job.')
    parser.add_argument('--wandb_group', default='group', type=str,
                        help='Group to put this experiment in on weights and biases.')


def train_args():

    parser = argparse.ArgumentParser()

    shared_args(parser)

    # Training / misc options.
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of epochs to train for.')
    parser.add_argument('--checkpoint_every', default=5, type=int,
                        help='Store permanent model checkpoint every this number of epochs.')
    parser.add_argument('--learn_rate', default=2e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_decay', default=0.3, type=float,
                        help='Learning rate factor per step for scheduler.')
    parser.add_argument('--do_val_aug', default=True, type=_str2bool,
                        help='If True, perform validation phase with data augmentation.')
    parser.add_argument('--do_val_noaug', default=False, type=_str2bool,
                        help='If True, also perform validation phase with no data augmentation '
                        'after every epoch, in addition to val_aug.')
    parser.add_argument('--val_every', default=2, type=int,
                        help='Epoch interval for validation phase(s).')
    
    # General data options.
    parser.add_argument('--num_frames', default=24, type=int,
                        help='Video clip length.')
    parser.add_argument('--image_height', default=192, type=int,
                        help='Vertical size of any entire post-processed image.')
    parser.add_argument('--image_width', default=256, type=int,
                        help='Horizontal size of any entire post-processed image.')

    # Model / architecture options.
    # ...

    # Loss & optimization options.
    parser.add_argument('--gradient_clip', default=0.5, type=float,
                        help='If > 0, clip gradient L2 norm to this value for stability.')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['sgd', 'adam', 'adamw', 'lamb'],
                        help='Which optimizer to use for training.')
    parser.add_argument('--l1_lw', default=1.0, type=float,
                        help='Weight for something.')
    
    # Ablations & baselines options.
    # ...

    args = parser.parse_args()
    verify_args(args, is_train=True)

    return args


def test_args():

    parser = argparse.ArgumentParser()

    # NOTE: Don't forget to consider this method as well when adding arguments.
    shared_args(parser)

    # Resource options.
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU index.')
    parser.add_argument('--num_batches', default=0, type=int,
                        help='If > 0, cut off script after this number of iterations, regardless '
                        'of dataset size or fraction, per given data path.')

    # Inference & processing options.
    parser.add_argument('--store_results', default=False, type=_str2bool,
                        help='In addition to generating lossy 2D visuals, save all inputs & '
                        'outputs to disk for later processing, visualizations, metrics, or other '
                        'deep dives.')
    parser.add_argument('--extra_visuals', default=0, type=int,
                        help='If > 0, generate and store extra visualizations.')

    # Automatically inferred options (do not assign).
    parser.add_argument('--test_log_path', default='', type=str,
                        help='Path to current logging directory for this experiment evaluation.')

    args = parser.parse_args()
    verify_args(args, is_train=False)

    return args


def verify_args(args, is_train=False):

    args.is_debug = args.name.startswith('d')

    args.wandb_group = ('train' if is_train else 'test') + \
        ('_debug' if args.is_debug else '')

    if args.num_workers < 0:
        if is_train:
            if args.is_debug:
                args.num_workers = max(int(mp.cpu_count() * 0.30) - 4, 4)
            else:
                args.num_workers = max(int(mp.cpu_count() * 0.45) - 6, 4)
        else:
            args.num_workers = max(mp.cpu_count() * 0.15 - 4, 4)
        args.num_workers = min(args.num_workers, 80)
    args.num_workers = int(args.num_workers)

    # If we have no name (e.g. for smaller scripts in eval), assume we are not interested in logging
    # either.
    if args.name != '':

        if args.resume != '':
            resume_name = args.resume
            if args.epoch >= 0:
                args.resume = os.path.join(args.checkpoint_root, args.resume, f'model_{args.epoch}.pth')
            else:
                args.resume = os.path.join(args.checkpoint_root, args.resume, 'checkpoint.pth')

        if is_train:
            # For example, --name v1.
            args.checkpoint_path = os.path.join(args.checkpoint_root, args.name)
            args.train_log_path = os.path.join(args.log_root, args.name)

            os.makedirs(args.checkpoint_path, exist_ok=True)
            os.makedirs(args.train_log_path, exist_ok=True)

            if args.resume != '':
                # Train example: --resume v3 --name dbg4.
                # NOTE: In this case, we wish to bootstrap another already trained model, yet resume
                # in our own new logs folder! The rest is handled by train.py.
                pass

            args.log_path = args.train_log_path

        else:
            assert args.resume != ''
            # Test example: --resume v1 --name t1.

            args.checkpoint_path = os.path.join(args.checkpoint_root, resume_name)
            args.train_log_path = os.path.join(args.log_root, resume_name)
            
            assert os.path.exists(args.checkpoint_path) and os.path.isdir(args.checkpoint_path)
            assert os.path.exists(args.train_log_path) and os.path.isdir(args.train_log_path)
            assert os.path.exists(args.resume) and os.path.isfile(args.resume)

            # Ensure that 0-based epoch is always part of the name and log directories.
            epoch = my_utils.get_checkpoint_epoch(args.resume)
            if args.perfect_baseline == 'none':
                args.name += f'_e{epoch}'
            else:
                args.name += f'_pb{args.perfect_baseline[:3]}'

            args.test_log_path = os.path.join(args.train_log_path, 'test_' + args.name)
            args.log_path = args.test_log_path
            os.makedirs(args.test_log_path, exist_ok=True)

    # NOTE: args.log_path is the one actually used by logvis.
