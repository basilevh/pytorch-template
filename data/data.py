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


def create_train_val_data_loaders(args, logger):
    '''
    return (train_loader, val_aug_loader, val_noaug_loader, dset_args).
    '''

    dset_args = dict()
    dset_args['image_height'] = 224
    dset_args['image_width'] = 288
    dset_args['use_data_frac'] = args.use_data_frac

    train_dataset = MyImageDataset(
        args.data_path, logger, 'train', do_random_augs=True, **dset_args)
    val_aug_dataset = MyImageDataset(
        args.data_path, logger, 'val', do_random_augs=True, **dset_args) \
        if args.do_val_aug else None
    val_noaug_dataset = MyImageDataset(
        args.data_path, logger, 'val', do_random_augs=False, **dset_args) \
        if args.do_val_noaug else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    val_aug_loader = torch.utils.data.DataLoader(
        val_aug_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False) \
        if args.do_val_aug else None
    val_noaug_loader = torch.utils.data.DataLoader(
        val_noaug_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False) \
        if args.do_val_noaug else None

    return (train_loader, val_aug_loader, val_noaug_loader, dset_args)


def create_test_data_loader(train_args, test_args, train_dset_args, logger):
    '''
    return (test_loader, test_dset_args).
    '''

    test_dset_args = copy.deepcopy(train_dset_args)
    test_dset_args['use_data_frac'] = test_args.use_data_frac

    test_dataset = MyImageDataset(
        test_args.data_path, logger, 'test', do_random_augs=False, **test_dset_args)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        shuffle=False, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)

    return (test_loader, test_dset_args)


class MyImageDataset(torch.utils.data.Dataset):
    '''
    PyTorch dataset class that returns uniformly random images of a labelled or unlabelled image
    dataset.
    '''

    def __init__(self, dataset_root, logger, phase, image_height=224, image_width=224,
                 use_data_frac=1.0, do_random_augs=False):
        '''
        :param dataset_root (str): Path to dataset (with or without phase).
        :param logger (MyLogger).
        :param phase (str): train / val_aug / val_noaug / test.
        :param transform: Data transform to apply on every image.
        '''
        self.log_info_fn = logger.info if logger is not None else print
        self.log_warning_fn = logger.warning if logger is not None else print

        # Get root and phase directories.
        phase_dp = os.path.join(dataset_root, phase)
        if not os.path.exists(phase_dp):
            # We may already be pointing to a phase directory (which, in the interest of
            # flexibility, is not necessarily the same as the passed phase argument).
            phase_dp = dataset_root
            dataset_root = str(pathlib.Path(dataset_root).parent)

        # Load all file paths beforehand.
        # NOTE: This method call handles subdirectories recursively, but also creates extra files.
        all_files = data_utils.cached_listdir(phase_dp, allow_exts=['jpg', 'jpeg', 'png'],
                                              recursive=True)
        file_count = len(all_files)
        self.log_info_fn(f'Image file count: {file_count}')
        dset_size = file_count

        # Define color and final resize transforms.
        to_tensor = torchvision.transforms.ToTensor()
        pre_transform = torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        post_transform = torchvision.transforms.Resize((image_height, image_width))

        self.dataset_root = dataset_root
        self.logger = logger
        self.phase = phase
        self.image_height = image_height
        self.image_width = image_width
        self.use_data_frac = use_data_frac
        self.do_random_augs = do_random_augs

        self.to_tensor = to_tensor
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.phase_dp = phase_dp
        self.all_files = all_files
        self.file_count = file_count
        self.dset_size = dset_size
        self.force_shuffle = (use_data_frac < 1.0)

    def __len__(self):
        return int(self.dset_size * self.use_data_frac)

    def __getitem__(self, index):
        # TODO: Select either deterministic or random mode.
        retries = 0
        file_idx = -1
        image_fp = ''

        if 0:
            # Read the image at the specified index.
            file_idx = index
            image_fp = self.all_files[file_idx]
            raw_image, _ = data_utils.read_image_robust(image_fp, no_fail=True)

        if 1:
            # Some files may be invalid, so we keep retrying (up to an upper bound).
            while True:
                try:

                    if not(self.force_shuffle) and retries == 0:
                        file_idx = index
                    else:
                        file_idx = np.random.randint(self.dset_size)

                    image_fp = self.all_files[file_idx]
                    raw_image, success = data_utils.read_image_robust(image_fp)
                    # (H, W, 3) array.

                    if success:
                        break

                except Exception as e:

                    self.log_warning_fn(f'retries: {retries}')
                    self.log_warning_fn(str(e))
                    self.log_warning_fn(f'image_fp: {image_fp}')
                    retries += 1
                    if retries >= 12:
                        raise e

        raw_image = self.to_tensor(raw_image / 255.0)  # (H, W, 3) array -> (3, H, W) tensor.
        raw_image = raw_image.type(torch.float32)
        (C, H, W) = raw_image.shape
        distort_image = raw_image

        if self.do_random_augs:
            crop_y1 = np.random.rand() * 0.1 + 0.1
            crop_y2 = np.random.rand() * 0.1 + 0.8
            crop_x1 = np.random.rand() * 0.1 + 0.1
            crop_x2 = np.random.rand() * 0.1 + 0.8
            crop_image = distort_image[:, int(crop_y1 * H):int(crop_y2 * H),
                                       int(crop_x1 * W):int(crop_x2 * W)]
            # Resize to final size (always).
            resize_image = self.post_transform(crop_image)

        else:
            resize_image = self.post_transform(distort_image)

        # Obtain ground truth.
        rgb_input = resize_image
        rgb_target = 1.0 - rgb_input

        # Return results.
        result = {'rgb_input': rgb_input,  # (3, H, W).
                  'rgb_target': rgb_target,  # (3, H, W).
                  'file_idx': file_idx,
                  'image_fp': image_fp}
        return result
