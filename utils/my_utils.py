'''
Miscellaneous tools / utilities / helper methods.
'''

from __init__ import *

# Library imports.
import io
import numpy as np
import os


def get_checkpoint_epoch(checkpoint_path):
    '''
    Gets the 0-based epoch index of a stored model checkpoint.
    '''
    epoch_path = checkpoint_path[:-4] + '_epoch.txt'

    if os.path.exists(epoch_path):
        epoch = int(np.loadtxt(epoch_path, dtype=np.int32))

    else:
        # NOTE: This backup method is inefficient but I'm not sure how to do it better.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint['epoch']

    return epoch


def any_value(my_dict):
    for (k, v) in my_dict.items():
        if v is not None:
            return v
    return None


def dict_to_cpu(x, ignore_keys=[]):
    '''
    Recursively converts all tensors in a dictionary to CPU.
    '''
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, dict):
        return {k: dict_to_cpu(v, ignore_keys=ignore_keys) for (k, v) in x.items()
        if not(k in ignore_keys)}
    elif isinstance(x, list):
        return [dict_to_cpu(v, ignore_keys=ignore_keys) for v in x]
    else:
        return x


def is_nan_or_inf(x):
    '''
    Returns True if x is NaN or Inf.
    '''
    if torch.is_tensor(x):
        return torch.isnan(x).any() or torch.isinf(x).any()
    else:
        return np.any(np.isnan(x)) or np.any(np.isinf(x))


def get_fourier_positional_encoding_size(num_coords, num_frequencies):
    '''
    Returns the embedding dimensionality of Fourier positionally encoded points.
    :param num_coords (int) = C: Number of coordinate values per point.
    :param num_frequencies (int) = F: Number of frequencies to use.
    '''
    return num_coords * (1 + num_frequencies * 2)  # Identity + (cos + sin) for each frequency.


def apply_fourier_positional_encoding(raw_coords, num_frequencies,
                                      base_frequency=0.1, max_frequency=10.0):
    '''
    Applies Fourier positional encoding (cos + sin) to a set of coordinates. Note that it is the
        caller's responsibility to manage the value ranges of all coordinate dimensions.
    :param raw_coords (*, C) tensor: Points with UVD or XYZ or other values.
    :param num_frequencies (int) = F: Number of frequencies to use.
    :param base_frequency (float) = f_0: Determines coarsest level of detail.
    :param max_frequency (int) = f_M: Determines finest level of detail.
    :return enc_coords (*,  C * (1 + F * 2)): Embedded points.
    '''
    assert num_frequencies > 0
    assert base_frequency > 0
    assert max_frequency > base_frequency

    enc_coords = []
    enc_coords.append(raw_coords.clone())

    for f in range(num_frequencies):
        cur_freq = f * (max_frequency - base_frequency) / (num_frequencies - 1) + base_frequency
        enc_coords.append((raw_coords * 2.0 * np.pi * cur_freq).cos())
        enc_coords.append((raw_coords * 2.0 * np.pi * cur_freq).sin())

    enc_coords = torch.cat(enc_coords, dim=-1)
    return enc_coords


def elitist_shuffle(items, inequality):
    '''
    https://github.com/rragundez/elitist-shuffle
    Shuffle array with bias over initial ranks
    A higher ranked content has a higher probability to end up higher
    ranked after the shuffle than an initially lower ranked one.
    Args:
        items (numpy.array): Items to be shuffled
        inequality (int/float): how biased you want the shuffle to be.
            A higher value will yield a lower probabilty of a higher initially
            ranked item to end up in a lower ranked position in the
            sequence.
    '''
    weights = np.power(
        np.linspace(1, 0, num=len(items), endpoint=False),
        inequality
    )
    weights = weights / np.linalg.norm(weights, ord=1)
    return np.random.choice(items, size=len(items), replace=False, p=weights)


def quick_pca(array, k=3, unique_features=False, normalize=None):
    '''
    array (*, n): Array to perform PCA on.
    k (int) < n: Number of components to keep.
    '''
    n = array.shape[-1]
    all_axes_except_last = tuple(range(len(array.shape) - 1))
    array_flat = array.reshape(-1, n)

    pca = sklearn.decomposition.PCA(n_components=k)
    
    if unique_features:
        # Obtain unique combinations of occluding instance sequences, to avoid bias toward larger
        # object masks.
        unique_combinations = np.unique(array_flat, axis=0)
        pca.fit(unique_combinations)
    
    else:
        pca.fit(array_flat)
    
    result_unnorm = pca.transform(array_flat).reshape(*array.shape[:-1], k)
    
    if normalize is not None:
        per_channel_min = result_unnorm.min(axis=all_axes_except_last, keepdims=True)
        per_channel_max = result_unnorm.max(axis=all_axes_except_last, keepdims=True)
        result = (result_unnorm - per_channel_min) / (per_channel_max - per_channel_min)
        result = result * (normalize[1] - normalize[0]) + normalize[0]

    else:
        result = result_unnorm
    
    result = result.astype(np.float32)
    return result


def ax_to_numpy(ax, dpi=160):
    fig = ax.figure
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw', dpi=dpi)
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.get_size_inches() * dpi
    image = data.reshape((int(h), int(w), -1))
    image = image[..., 0:3] / 255.0
    return image


def disk_cached_call(logger, cache_fp, newer_than, func, *args, **kwargs):
    '''
    Caches the result of a function call to disk, to avoid recomputing it.
    :param cache_fp (str): Path to cache file.
    :param newer_than (float): Cache must be more recent than this UNIX timestamp.
    :param func (callable): Function to call.
    :param args: Positional arguments to pass to function.
    :param kwargs: Keyword arguments to pass to function.
    :return: Result of function call.
    '''
    use_cache = (cache_fp is not None and os.path.exists(cache_fp))
    if use_cache and newer_than is not None and os.path.getmtime(cache_fp) < newer_than:
        logger.info(f'Deleting too old cached result at {cache_fp}...')
        use_cache = False
        os.remove(cache_fp)

    if use_cache:
        # logger.debug(f'Loading cached {func.__name__} result from {cache_fp}...')
        with open(cache_fp, 'rb') as f:
            result = pickle.load(f)

    else:
        result = func(*args, **kwargs)
        
        if cache_fp is not None:
            cache_dp = str(pathlib.Path(cache_fp).parent)
            os.makedirs(cache_dp, exist_ok=True)
            logger.info(f'Caching {func.__name__} result to {cache_fp}...')
            with open(cache_fp, 'wb') as f:
                pickle.dump(result, f)

    return result


def read_txt_strip_comments(txt_fp):
    with open(txt_fp, 'r') as f:
        lines = f.readlines()
    lines = [x.split('#')[0] for x in lines]
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if len(x) > 0]
    return lines


if __name__ == '__main__':

    torch.set_printoptions(precision=3, sci_mode=False)
