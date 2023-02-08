'''
Data loading and processing logic.
Created by Basile Van Hoorick.
'''

from __init__ import *

# Internal imports.
import data_2d
import data_utils


def _seed_worker(worker_id):
    '''
    Ensures that every data loader worker has a separate seed with respect to NumPy and Python
    function calls, not just within the torch framework. This is very important as it sidesteps
    lack of randomness- and augmentation-related bugs.
    '''
    worker_seed = torch.initial_seed() % (2 ** 32)  # This is distinct for every worker.
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)


def create_train_val_data_loaders(args, logger):
    '''
    return (train_loader, val_aug_loader, val_noaug_loader, dset_args).
    '''
    actual_data_paths = args.data_path
    assert isinstance(actual_data_paths, list)

    cur_data_path = actual_data_paths[0]

    dset_args = dict()
    dset_args['use_data_frac'] = args.use_data_frac

    tf_aug = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomResizedCrop((args.image_height, args.image_width)),
    ])
    tf_noaug = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((args.image_height, args.image_width)),
    ])

    train_dset = data_2d.MySimpleImageDataset(
        cur_data_path, logger=logger, phase_dn='train', processor=tf_aug, **dset_args)
    val_aug_dset = data_2d.MySimpleImageDataset(
        cur_data_path, logger=logger, phase_dn='val', processor=tf_aug, **dset_args) \
        if args.do_val_aug else None
    val_noaug_dset = data_2d.MySimpleImageDataset(
        cur_data_path, logger=logger, phase_dn='val', processor=tf_noaug, **dset_args) \
        if args.do_val_noaug else None

    shuffle = True
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=shuffle, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    val_aug_loader = torch.utils.data.DataLoader(
        val_aug_dset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=shuffle, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False) \
        if args.do_val_aug else None
    val_noaug_loader = torch.utils.data.DataLoader(
        val_noaug_dset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=shuffle, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False) \
        if args.do_val_noaug else None

    return (train_loader, val_aug_loader, val_noaug_loader, dset_args)


def create_test_data_loader(train_args, test_args, train_dset_args, logger):
    '''
    return (test_loader, test_dset_args).
    '''
    actual_data_paths = test_args.data_path
    assert isinstance(actual_data_paths, list)

    cur_data_path = test_args.data_path[0]

    test_dset_args = copy.deepcopy(train_dset_args)
    test_dset_args['use_data_frac'] = test_args.use_data_frac

    tf_noaug = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((train_args.image_height, train_args.image_width)),
    ])

    test_dset = data_2d.MySimpleImageDataset(
        cur_data_path, logger=logger, phase_dn='test', processor=tf_noaug, **test_dset_args)

    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        shuffle=False, worker_init_fn=_seed_worker, drop_last=False, pin_memory=False)

    return (test_loader, test_dset_args)
