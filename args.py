'''
Handling of parameters that can be passed to training and testing scripts.
'''

from __init__ import *


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
    parser.add_argument('--seed', default=2022, type=int,
                        help='Random number generator seed.')

    # Resource options.
    parser.add_argument('--device', default='cuda', type=str,
                        help='cuda or cpu.')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of data loading workers; -1 means automatic.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size during training or testing.')

    # Logging & checkpointing options.
    parser.add_argument('--data_path', default='/path/to/datasets/ImageNet/', type=str,
                        help='Path to dataset root folder.')
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
                        
    # Data options (all phases).
    parser.add_argument('--use_data_frac', default=0.1, type=float,
                        help='If < 1.0, use a smaller dataset.')

    # Automatically inferred options (do not assign).
    parser.add_argument('--is_debug', default=False, type=_str2bool,
                        help='Shorter epochs; log and visualize more often.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='Path to current checkpoint directory for this experiment.')
    parser.add_argument('--train_log_path', default='', type=str,
                        help='Path to current train-time logging directory for this experiment.')
    parser.add_argument('--log_path', default='', type=str,
                        help='Switches to train or test depending on the job.')


def train_args():

    parser = argparse.ArgumentParser()

    shared_args(parser)

    # Training / misc options.
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of epochs to train for.')
    parser.add_argument('--checkpoint_every', default=5, type=int,
                        help='Store permanent model checkpoint every this number of epochs.')
    parser.add_argument('--learn_rate', default=5e-4, type=float,
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
    parser.add_argument('--gradient_clip', default=0.4, type=float,
                        help='If > 0, clip gradient L2 norm to this value for stability.')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        help='Which optimizer to use for training (sgd / adam / adamw / lamb).')

    # Model options.
    parser.add_argument('--image_height', default=224, type=int,
                        help='Vertical size of any entire image after data transforms.')
    parser.add_argument('--image_width', default=288, type=int,
                        help='Horizontal size of any entire image after data transforms.')
    
    # Loss options.
    parser.add_argument('--l1_lw', default=0.5, type=float,
                        help='Weight for L1 pixel reconstruction loss terms.')

    args = parser.parse_args()
    verify_args(args, is_train=True)

    return args


def test_args():

    parser = argparse.ArgumentParser()

    shared_args(parser)

    # Resource options.
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU index.')

    # Automatically inferred options (do not assign).
    parser.add_argument('--test_log_path', default='', type=str,
                        help='Path to current logging directory for this experiment evaluation.')

    args = parser.parse_args()
    verify_args(args, is_train=False)

    return args


def verify_args(args, is_train=False):

    assert args.device in ['cuda', 'cpu']

    args.is_debug = args.name.startswith('d')

    if is_train:

        # Handle allowable options.
        assert args.optimizer in ['sgd', 'adam', 'adamw', 'lamb']

    if args.num_workers < 0:
        if is_train:
            if args.is_debug:
                args.num_workers = max(int(mp.cpu_count() * 0.45) - 6, 4)
            else:
                args.num_workers = max(int(mp.cpu_count() * 0.95) - 8, 4)
        else:
            args.num_workers = max(mp.cpu_count() * 0.25 - 4, 4)
        args.num_workers = min(args.num_workers, 116)
    args.num_workers = int(args.num_workers)

    # If we have no name (e.g. for smaller scripts in eval), assume we are not interested in logging
    # either.
    if args.name != '':

        if is_train:
            # For example, --name v1.
            args.checkpoint_path = os.path.join(args.checkpoint_root, args.name)
            args.train_log_path = os.path.join(args.log_root, args.name)

            os.makedirs(args.checkpoint_path, exist_ok=True)
            os.makedirs(args.train_log_path, exist_ok=True)

        if args.resume != '':
            # Train example: --resume v3 --name dbg4.
            # Test example: --resume v1 --name t1.
            # NOTE: In case of train, --name will mostly be ignored.
            args.checkpoint_path = os.path.join(args.checkpoint_root, args.resume)
            args.train_log_path = os.path.join(args.log_root, args.resume)

            if args.epoch >= 0:
                args.resume = os.path.join(args.checkpoint_path, f'model_{args.epoch}.pth')
                args.name += f'_e{args.epoch}'
            else:
                args.resume = os.path.join(args.checkpoint_path, 'checkpoint.pth')

            assert os.path.exists(args.checkpoint_path) and os.path.isdir(args.checkpoint_path)
            assert os.path.exists(args.train_log_path) and os.path.isdir(args.train_log_path)
            assert os.path.exists(args.resume) and os.path.isfile(args.resume)

        if not(is_train):
            assert args.resume != ''
            args.test_log_path = os.path.join(args.train_log_path, 'test_' + args.name)
            args.log_path = args.test_log_path
            os.makedirs(args.test_log_path, exist_ok=True)

        else:
            args.log_path = args.train_log_path
