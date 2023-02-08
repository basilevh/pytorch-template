'''
Data loading and processing logic.
Created by Basile Van Hoorick.
'''

from __init__ import *

# Library imports.
from PIL import Image

# Internal imports.
import data_utils


class MySimpleImageDataset(torch.utils.data.Dataset):
    '''
    PyTorch dataset class that returns unlabelled images of any image dataset directory in order of
        sorted file paths. Preprocessing / transforms / augmentations are left to the caller.
    '''

    def __init__(self, dset_root, logger=None, phase_dn=None, processor=None, use_data_frac=1.0,
                 index_handling='order'):
        '''
        :param dset_root (str): Path to dataset.
            All images located somewhere (recursively) inside this directory will be loaded.
        :param phase_dn (str): If given, subdirectory to look for.
        :param processor (torchvision.transforms): If given, will be applied to each PIL image.
        :param use_data_frac (float) in [0, 1].
        :param index_handling (str): order / random.
            If order, choose deterministically and use selected files in ascending order only.
            If data_frac < 1, this implies that the rest of the data will remain unused forever.
            If random, choose uniformly randomly among all available files at every step.
            If data_frac < 1, this implies that eventually more data will get used across multiple
            epochs.
        '''
        self.info_fn = print if logger is None else logger.info
        self.warn_fn = print if logger is None else logger.warning
        self.error_fn = print if logger is None else logger.error

        # Fix options.
        if phase_dn is not None and os.path.exists(os.path.join(dset_root, phase_dn)):
            dset_root = os.path.join(dset_root, phase_dn)
        if processor is None:
            processor = torchvision.transforms.ToTensor()

        # Assign variables.
        self.dset_root = dset_root
        self.logger = logger
        self.phase_dn = phase_dn
        self.processor = processor
        self.use_data_frac = use_data_frac
        self.index_handling = index_handling

        # Load all available image file paths recursively beforehand.
        if os.path.isdir(self.dset_root):
            image_fps = data_utils.recursive_listdir(
                self.dset_root, extensions=['jpg', 'jpeg', 'png'])
            self.image_fps = sorted(image_fps)
            self.num_avail_images = len(self.image_fps)

        else:
            # Fill entire batch with just a single provided file.
            self.image_fps = [self.dset_root] * 256
            self.num_avail_images = 256
            self.warn_fn(f'Only a single file is provided: {self.dset_root}')

        # Choose smaller subset of discovered data if applicable.
        self.used_dset_size = int(self.num_avail_images * self.use_data_frac)
        self.info_fn('Available image files:', self.num_avail_images)
        self.info_fn('Used dataset size:', self.used_dset_size)

    def __len__(self):
        return self.used_dset_size

    def __getitem__(self, index):
        '''
        :param index (int).
        :return data_retval (dict).
        '''
        if self.index_handling == 'order':
            file_idx = index % self.num_avail_images
        
        elif self.index_handling == 'random':
            file_idx = np.random.randint(0, self.num_avail_images)

        else:
            raise ValueError(f'Unknown index_handling: {self.index_handling}')

        src_fp = self.image_fps[file_idx]

        try:
            raw_image = Image.open(src_fp).convert('RGB')
            rgb = self.processor(raw_image).unsqueeze(0)

        except Exception as e:
            self.error_fn(f'Exception while loading or preprocessing image file {src_fp}: {e}')
            raise e

        data_retval = dict()
        data_retval['dset_idx'] = index
        data_retval['file_idx'] = file_idx
        data_retval['src_fp'] = src_fp
        data_retval['rgb'] = rgb
        
        return data_retval
