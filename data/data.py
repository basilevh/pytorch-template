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

    dset_args = dict()
    dset_args['image_height'] = args.image_height
    dset_args['image_width'] = args.image_width
    dset_args['use_data_frac'] = args.use_data_frac

    cur_data_path = actual_data_paths[0]

    train_dset = MyImageDataset(
        cur_data_path, logger, 'train', **dset_args)
    val_aug_dset = MyImageDataset(
        cur_data_path, logger, 'val_aug', **dset_args) if args.do_val_aug else None
    val_noaug_dset = MyImageDataset(
        cur_data_path, logger, 'val_noaug', **dset_args) if args.do_val_noaug else None
    
    dset_args = dict()

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

    def __init__(self, dset_root, logger, phase, image_height=224, image_width=224,
                 use_data_frac=1.0):
        '''
        :param dset_root (str): Path to dataset (with or without phase).
        :param logger (MyLogger).
        :param phase (str): train / val_aug / val_noaug / test.
        '''
        self.dset_root = dset_root
        self.logger = logger
        self.phase = phase
        self.use_data_frac = use_data_frac

        # Image / frame options.
        self.image_height = image_height
        self.image_width = image_width
        
        # Change whether to apply random color jittering, flipping, and cropping.
        self.do_random_augs = (('train' in phase or 'val' in phase) and not('noaug' in phase))
        self.to_tensor = torchvision.transforms.ToTensor()

        # Get phase name with respect to file system.
        if 'train' in phase:
            phase_dn = 'train'
        elif 'val' in phase:
            phase_dn = 'val'
        elif 'test' in phase:
            phase_dn = 'test'
        else:
            raise ValueError(phase)
        
        # Get actual root and phase directories.
        phase_dp = os.path.join(dset_root, phase_dn)
        if not os.path.exists(phase_dp):
            phase_dp = dset_root

        # Load all file paths beforehand.
        # NOTE: This method call handles subdirectories recursively, but also creates extra files.
        image_fps = data_utils.cached_listdir(phase_dp, allow_exts=['jpg', 'jpeg', 'png'],
                    recursive=True)
        num_images = len(image_fps)

        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        manager = mp.Manager()
        scene_dps = manager.list(scene_dps)

        # Instantiate custom augmentation pipeline.
        # Define color and final resize transforms.
        pre_transform = torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.06)
        post_transform = torchvision.transforms.Resize((image_height, image_width))
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        # Assign variables.
        self.phase_dn = phase_dn
        self.phase_dp = phase_dp
        self.image_fps = image_fps
        self.dset_size = num_images
        self.used_dset_size = int(use_data_frac * num_images)
        self.force_shuffle = (use_data_frac < 1.0 and ('train' in phase or 'val' in phase))

        self.logger.info(f'(MyImageDataset) ({phase}) File count: {num_images}')
        self.logger.info(f'(MyImageDataset) ({phase}) Used dataset size: {self.used_dset_size}')

    def __len__(self):
        return self.used_dset_size

    def __getitem__(self, index):
        '''
        :return data_retval (dict).
        '''
        # Some files / scenes may be invalid, so we keep retrying (up to an upper bound).
        retries = 0
        file_idx = -1

        while True:
            try:
                if not(self.force_shuffle) and retries == 0:
                    file_idx = index % self.dset_size
                else:
                    file_idx = np.random.randint(self.dset_size)

                image_fp = copy.deepcopy(self.image_fps[file_idx])
                data_retval = self._load_example(file_idx, image_fp)

                break  # We are successful if we reach this.

            except Exception as e:
                retries += 1
                self.log_warning_fn(f'(MyImageDataset) file_idx / image_fp: {file_idx} / {image_fp}')
                self.log_warning_fn(f'(MyImageDataset) {str(e)}')
                self.log_warning_fn(f'(MyImageDataset) retries: {retries}')
                if retries >= 12:
                    raise e
        
        # Pass on metadata.
        data_retval['retries'] = retries
        data_retval['file_idx'] = file_idx
        data_retval['image_fp'] = image_fp

        return data_retval


    def _load_example(self, file_idx, image_fp):
        '''
        :return data_retval (dict).
        '''
        (raw_image, success) = data_utils.read_image_robust(image_fp)
        if not(success):
            raise RuntimeError(f'Failed to properly load image: {image_fp}')

        raw_image = self.to_tensor(raw_image / 255.0)  # (H, W, 3) array -> (3, H, W) tensor.
        raw_image = raw_image.type(torch.float32)
        (C, H, W) = raw_image.shape
        distort_image = raw_image

        if self.do_random_augs:
            # Apply color perturbation (train time only).
            distort_image = self.pre_transform(distort_image)

            # Apply crops (train time only).
            crop_y1 = np.random.rand() * 0.1 + 0.1
            crop_y2 = np.random.rand() * 0.1 + 0.8
            crop_x1 = np.random.rand() * 0.1 + 0.1
            crop_x2 = np.random.rand() * 0.1 + 0.8
            crop_image = distort_image[:, int(crop_y1 * H):int(crop_y2 * H),
                                       int(crop_x1 * W):int(crop_x2 * W)]

        else:
            crop_image = distort_image

        # Resize to final size (always).
        resize_image = self.post_transform(crop_image)

        # Obtain ground truth.
        rgb_input = resize_image
        rgb_target = 1.0 - rgb_input

        # Organize & return results.
        data_retval = dict()
        data_retval['rgb_input'] = rgb_input  # (3, H, W).
        data_retval['rgb_target'] = rgb_target  # (3, H, W).

        return raw_image
