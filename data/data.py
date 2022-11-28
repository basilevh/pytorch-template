'''
Data loading and processing logic.
'''

from __init__ import *

# Internal imports.
import augs
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

    # TODO: Figure out noaug val dataset args as well.
    dset_args = dict()
    dset_args['use_data_frac'] = args.use_data_frac

    cur_data_path = actual_data_paths[0]

    train_dset = MyImageDataset(
        cur_data_path, logger, 'train', **dset_args)
    val_aug_dset = MyImageDataset(
        cur_data_path, logger, 'val', **dset_args) \
        if args.do_val_aug else None
    val_noaug_dset = MyImageDataset(
        cur_data_path, logger, 'val', **dset_args) \
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

    # Due to the nature of testing, we will simply use ConcatDataset instead of MixedDataset here
    # when there are multiple data sources.
    # NOTE: Only the last test_dset_args of each source in the list is remembered and returned.
    # test_dset_list = []

    cur_data_path = test_args.data_path[0]

    test_dset_args = copy.deepcopy(train_dset_args)
    test_dset_args['use_data_frac'] = test_args.use_data_frac

    test_dset = MyImageDataset(
        cur_data_path, logger, 'test', **test_dset_args)

    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        shuffle=False, worker_init_fn=_seed_worker, drop_last=False, pin_memory=False)

    return (test_loader, test_dset_args)


class MyImageDataset(torch.utils.data.Dataset):
    '''
    PyTorch dataset class that returns uniformly random images of a labelled or unlabelled image
    dataset.
    '''

    def __init__(self, dataset_root, logger, phase, use_data_frac=1.0):
        '''
        :param dataset_root (str): Path to dataset (with or without phase).
        :param logger (MyLogger).
        :param phase (str): train / val_aug / val_noaug / test.
        '''
        # Get root and phase directories.
        phase_dir = os.path.join(dataset_root, phase)
        if not os.path.exists(phase_dir):
            # We may already be pointing to a phase directory (which, in the interest of
            # flexibility, is not necessarily the same as the passed phase argument).
            phase_dir = dataset_root
            dataset_root = str(pathlib.Path(dataset_root).parent)

        # Load all file paths beforehand.
        # NOTE: This method call handles subdirectories recursively, but also creates extra files.
        all_files = data_utils.cached_listdir(phase_dir, allow_exts=['jpg', 'jpeg', 'png'],
                                              recursive=True)
        file_count = len(all_files)
        print('Image file count:', file_count)
        dset_size = file_count

        self.dataset_root = dataset_root
        self.logger = logger
        self.phase = phase
        self.phase_dir = phase_dir
        self.transform = None
        self.all_files = all_files
        self.file_count = file_count
        self.dset_size = dset_size

    def __len__(self):
        return self.dset_size

    def __getitem__(self, index):
        # TODO: Select either deterministic or random mode.
        # Sometimes, not every element in the dataset is actually suitable, in which case retries
        # may be needed, and as such the latter option is preferred.

        if 1:
            # Read the image at the specified index.
            file_idx = index
            image_fp = self.all_files[file_idx]
            rgb_input, _ = data_utils.read_image_robust(image_fp, no_fail=True)

        if 0:
            # Read a random image.
            success = True
            file_idx = -1
            while not success:
                file_idx = np.random.choice(self.file_count)
                image_fp = self.all_files[file_idx]
                rgb_input, success = data_utils.read_image_robust(image_fp)

        # Apply transforms.
        if self.transform is not None:
            rgb_input = self.transform(rgb_input)

        # Obtain ground truth.
        rgb_target = 1.0 - rgb_input

        # Return results.
        result = {'rgb_input': rgb_input,  # (H, W, 3).
                  'rgb_target': rgb_target,  # (H, W, 3).
                  'file_idx': file_idx,
                  'path': image_fp}
        return result
