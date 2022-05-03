'''
Data-related utilities and helper methods.
'''

from __init__ import *


def cached_listdir(dir_path, allow_exts=[], recursive=False, force_live=False):
    '''
    Returns a list of all file paths if needed, and caches the result for efficiency.
    NOTE: Manual deletion of listdir.p is required if the directory ever changes.
    :param dir_path (str): Folder to gather file paths within.
    :param allow_exts (list of str): Only retain files matching these extensions.
    :param recursive (bool): Also include contents of all subdirectories within.
    :param force_live (bool): Same behavior but don't read or write any cache.
    :return (list of str): List of full image file paths.
    '''
    exts_str = '_'.join(allow_exts)
    recursive_str = 'rec' if recursive else ''
    cache_fp = f'{str(pathlib.Path(dir_path))}_{exts_str}_{recursive_str}_cld.p'

    if os.path.exists(cache_fp) and not(force_live):
        # Cached result already available.
        print('Loading directory contents from ' + cache_fp + '...')
        with open(cache_fp, 'rb') as f:
            result = pickle.load(f)

    else:
        # No cached result available yet. This call can sometimes be very expensive.
        raw_listdir = os.listdir(dir_path)
        result = copy.deepcopy(raw_listdir)

        # Append root directory to get full paths.
        result = [os.path.join(dir_path, fn) for fn in result]
        
        # Filter by files only (no folders), not being own cache dump,
        # belonging to allowed file extensions, and not starting with a dot.
        result = [fp for fp in result if os.path.isfile(fp)]
        result = [fp for fp in result if not fp.endswith('_cld.p')]
        if allow_exts is not None and len(allow_exts) != 0:
            result = [fp for fp in result
                      if any([fp.lower().endswith('.' + ext) for ext in allow_exts])
                      and fp[0] != '.']

        # Recursively append contents of subdirectories within.
        if recursive:
            for dn in raw_listdir:
                dp = os.path.join(dir_path, dn)
                if os.path.isdir(dp):
                    result += cached_listdir(dp, allow_exts=allow_exts, recursive=True)
                  
        if not(force_live):
            print('Caching filtered directory contents to ' + cache_fp + '...')
            with open(cache_fp, 'wb') as f:
                pickle.dump(result, f)
    
    return result


def read_image_robust(img_path, no_fail=False):
    '''
    Loads and returns an image that meets conditions along with a success flag, in order to avoid
    crashing.
    '''
    try:
        image = plt.imread(img_path).copy()
        success = True
        if (image.ndim != 3 or image.shape[2] != 3
                or np.any(np.array(image.strides) < 0)):
            # Either not RGB or has negative stride, so discard.
            success = False
            if no_fail:
                raise RuntimeError(f'ndim: {image.ndim}  '
                                   f'shape: {image.shape}  '
                                   f'strides: {image.strides}')

    except IOError as e:
        # Probably corrupt file.
        image = None
        success = False
        if no_fail:
            raise e

    return image, success
