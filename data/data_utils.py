'''
Data-related utilities and helper methods.
Created by Basile Van Hoorick.
'''

from __init__ import *

# Library imports.
import glob
import joblib

# Internal imports.
import my_utils

cache = joblib.Memory('cache/', verbose=1)


@cache.cache
def recursive_listdir(src_dp, extensions=['jpg', 'jpeg', 'png']):
    '''
    :param src_dp (str).
    :param extensions (list of str).
    :return src_fps (list of str).
    '''
    src_fps = []
    
    # Get all full file paths with specified extensions.
    for ext in extensions:
        src_fps += glob.glob(os.path.join(src_dp, '**/*.' + ext.lower()), recursive=True)
        src_fps += glob.glob(os.path.join(src_dp, '**/*.' + ext.upper()), recursive=True)
    src_fps = sorted(src_fps)

    # Exclude probably invalid temporary file names starting with a dot.
    src_fps = [fp for fp in src_fps if not(fp.split('/')[-1].startswith('.'))]

    return src_fps


def read_video_audio_clip(video_fp, audio_fp, sel_num_frames=16, sel_start_time=None,
                          read_audio=True):
    # https://pytorch.org/vision/main/auto_examples/plot_video_api.html
    # https://pytorch.org/vision/stable/io.html#video
    # NOTE: this doesn't work with pixel phone recorded videos.
    av_reader = torchvision.io.VideoReader(video_fp, 'video')
    metadata = copy.deepcopy(av_reader.get_metadata())

    video_duration = metadata['video']['duration'][0]
    video_fps = metadata['video']['fps'][0]
    assert video_duration > 0
    assert video_fps > 0

    # Set clip boundaries.
    available_frames = round(video_duration * video_fps)
    assert available_frames >= 4
    if sel_num_frames <= 0:
        sel_num_frames = available_frames + 1  # This ensures counter never breaks.
    if sel_start_time == 'random':
        # Max end value leaves a small margin to ensure audio synchronization is always possible.
        start_frame = np.random.randint(0, available_frames - sel_num_frames - 2)
    else:
        # Round to ensure the requested timestamp (in seconds) is followed accurately.
        start_frame = int(np.round(sel_start_time * video_fps))
    start_time = start_frame / video_fps
    end_frame = start_frame + sel_num_frames
    end_time = end_frame / video_fps

    # Read clip video frames always.
    av_reader.set_current_stream('video')
    av_reader.seek(start_time)
    frames = []
    video_pts = []
    counter = 0

    for frame in av_reader:
        frames.append(frame['data'])  # (C, H, W) tensor.
        video_pts.append(frame['pts'])  # single float.
        counter += 1
        if counter >= sel_num_frames:
            break

    frames = torch.stack(frames, 0)  # (T, C, H, W) tensor.
    video_real_start = video_pts[0]
    video_real_end = video_pts[-1] + 1.0 / video_fps

    # NOTE: For these requested values to be correct, we assume video_pts is equally spaced with
    # interval = 1 / FPS. Other cases are probably rare and difficult to deal with.
    metadata['vid_avail_frames'] = available_frames
    metadata['vid_duration'] = video_duration
    metadata['vid_fps'] = video_fps
    metadata['vid_start_frame'] = start_frame
    metadata['vid_start_time'] = start_time
    metadata['vid_end_frame'] = end_frame
    metadata['vid_end_time'] = end_time
    metadata['vid_real_start'] = video_real_start
    metadata['vid_real_end'] = video_real_end

    # Read clip audio waveform if desired.
    if read_audio:
        if audio_fp is not None:
            av_reader = torchvision.io.VideoReader(audio_fp, 'audio')
            audio_metadata = copy.deepcopy(av_reader.get_metadata())
            metadata['audio'] = audio_metadata['audio']

        audio_duration = metadata['audio']['duration'][0]
        audio_sample_rate = metadata['audio']['framerate'][0]
        assert audio_duration > 0
        assert audio_sample_rate > 0
        assert audio_duration >= video_duration

        # NOTE: We assume audio packets are always shorter than video frames, and that PTS
        # (presentation timestamp) denotes when to _start_ presenting a packet.
        # https://en.wikipedia.org/wiki/Presentation_timestamp
        av_reader.set_current_stream('audio')
        av_reader.seek(video_real_start - 1.0 / video_fps)

        wavelets = []
        audio_pts = []
        counter = 0

        for frame in av_reader:
            wavelets.append(frame['data'])  # (T, C) tensor with WXYZ if C = 4.
            audio_pts.append(frame['pts'])  # single float.
            counter += 1
            if audio_pts[-1] > video_real_end:
                break

        waveform = torch.cat(wavelets, 0)  # (T, C) tensor with WXYZ if C = 4.
        audio_real_start = audio_pts[0]
        audio_real_end = audio_pts[-1] + wavelets[-1].shape[0] / audio_sample_rate

        # NOTE: video_pts and audio_pts have different temporal strides / intervals, i.e.
        # (1 / video_fps) and (wavelet_size / sample_rate) respectively. We sampled audio packets in
        # such a way that it encompasses the video clip, so now we simply concatenate and mark the
        # appropriate subset to take.
        num_samples = waveform.shape[0]
        align_start = (video_real_start - audio_real_start) / (audio_real_end - audio_real_start)
        align_end = (video_real_end - audio_real_start) / (audio_real_end - audio_real_start)
        align_start_idx = int(align_start * num_samples)
        align_end_idx = int(align_end * num_samples)

        waveform = waveform[align_start_idx:align_end_idx]
        num_samples_per_frame = audio_sample_rate / video_fps
        nspf_align = waveform.shape[0] / frames.shape[0]
        np.testing.assert_approx_equal(nspf_align, num_samples_per_frame, significant=3)

        metadata['aud_duration'] = audio_duration
        metadata['aud_sample_rate'] = audio_sample_rate
        metadata['aud_read_samples'] = num_samples
        metadata['aud_read_real_start'] = audio_real_start
        metadata['aud_read_real_end'] = audio_real_end
        metadata['aud_align_start_idx'] = align_start_idx
        metadata['aud_align_end_idx'] = align_end_idx
        metadata['aud_samples_per_frame'] = num_samples_per_frame

    else:
        waveform = None
        audio_pts = None

    del av_reader
    del metadata['video']
    del metadata['audio']

    return (frames, waveform, video_pts, audio_pts, metadata)


def read_all_images(src_dp, exclude_patterns=None, count_only=False, use_tqdm=False, stack=False,
                    early_resize_height=None):
    '''
    :param src_dp (str).
    :return frames (T, H, W, 3) array with float32 in [0, 1].
    '''
    src_fps = list(sorted(glob.glob(os.path.join(src_dp, '*.jpg')) +
                          glob.glob(os.path.join(src_dp, '*.png'))))

    if count_only:
        return len(src_fps)

    if exclude_patterns is not None:
        if not(isinstance(exclude_patterns, list)):
            exclude_patterns = [exclude_patterns]
        for pattern in exclude_patterns:
            src_fps = [fp for fp in src_fps if not(pattern in fp)]

    frames = []
    if use_tqdm:
        src_fps = tqdm.tqdm(src_fps)

    for fp in src_fps:
        frame = plt.imread(fp)[..., 0:3]
        frame = (frame / 255.0).astype(np.float32)

        if early_resize_height is not None and early_resize_height > 0:
            (H1, W1) = frame.shape[:2]
            if H1 > early_resize_height:
                (H2, W2) = (early_resize_height, int(round(early_resize_height * W1 / H1)))
                frame = cv2.resize(frame, (W2, H2), interpolation=cv2.INTER_LINEAR)

        frames.append(frame)

    if stack:  # Otherwise, remain list for efficiency.
        frames = np.stack(frames)

    return frames


def read_image_robust(img_path, no_fail=False):
    '''
    Loads and returns an image that meets conditions along with a success flag, in order to avoid
    crashing.
    '''
    try:
        image = plt.imread(img_path).copy()  # (H, W) or (H, W, 3) array of uint8.
        success = True

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.stack([image[..., 0]] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., 0:3]

        if (image.ndim != 3 or image.shape[2] != 3
                or np.any(np.array(image.strides) < 0)):
            # Either not RGB or has negative stride, so discard.
            success = False
            if no_fail:
                raise RuntimeError(f'ndim: {image.ndim}  '
                                   f'dtype: {image.dtype}  '
                                   f'shape: {image.shape}  '
                                   f'strides: {image.strides}')

    except IOError as e:
        # Probably corrupt file.
        image = None
        success = False
        if no_fail:
            raise e

    return (image, success)


def pad_div_numpy(div_array, axes, max_size):
    '''
    Adds zeros to the first axis of an array such that it can be collated.
    :param div_array (K, *) array.
    :param axes (tuple) of int.
    :param max_size (int) = M.
    :return (padded_div_array, K).
        padded_div_array (M, *) array.
        K (int).
    '''
    K = -1
    pad_width = [(0, 0) for _ in range(div_array.ndim)]

    for axis in axes:
        cur_K = div_array.shape[axis]
        if K == -1:
            K = cur_K
        else:
            assert cur_K == K

        pad_width[axis] = (0, max_size - K)

    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    padded_div_array = np.pad(div_array, pad_width, mode='constant', constant_values=0)

    return (padded_div_array, K)


def pad_div_torch(div_tensor, axes, max_size):
    '''
    Adds zeros to the first axis of a tensor such that it can be collated.
    :param div_tensor (K, *) tensor.
    :param axes (tuple) of int.
    :param max_size (int) = M.
    :return (padded_div_tensor, K).
        padded_div_tensor (M, *) tensor.
        K (int).
    '''
    K = -1
    pad_width = [(0, 0) for _ in range(div_tensor.ndim)]

    for axis in axes:
        cur_K = div_tensor.shape[axis]
        if K == -1:
            K = cur_K
        else:
            assert cur_K == K

        pad_width[axis] = (0, max_size - K)

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    pad_width = list(np.array(list(reversed(pad_width))).flatten())
    padded_div_tensor = torch.nn.functional.pad(div_tensor, pad_width, mode='constant', value=0)

    return (padded_div_tensor, K)


def clean_remain_reproducible(data_retval):
    '''
    Prunes a returned batch of examples such that it can be reconstructed deterministically.
        This is useful to save space for debugging, evaluation, and visualization, because
        data_retval can be huge.
    '''
    data_retval_pruned = my_utils.dict_to_cpu(
        data_retval, ignore_keys=['todo'])

    return data_retval_pruned


def _paths_from_txt(txt_fp):
    # First, simply obtain all non-empty, non-commented lines.
    with open(txt_fp, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    lines = [line for line in lines if not(line.lower().startswith('#'))]
    
    # Then, prepend non-existent (presumed relative) paths with the directory of the text file.
    # This allows for sharing text files across machines without having to modify the contents.
    txt_dp = str(pathlib.Path(txt_fp).parent)
    paths = []
    for line in lines:
        if os.path.exists(line):
            paths.append(line)
        else:
            absolute_path = os.path.join(txt_dp, line)
            assert os.path.exists(absolute_path), absolute_path
            paths.append(absolute_path)
    
    return paths


def get_data_paths_from_args(given_data_paths):
    '''
    Converts any text file into the the list of actual paths it contains as lines within.
    '''
    actual_data_paths = []
    for data_path in given_data_paths:
        if data_path.lower().endswith('.txt'):
            actual_data_paths += _paths_from_txt(data_path)
        else:
            actual_data_paths.append(data_path)
    return actual_data_paths
