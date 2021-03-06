'''
Logging and visualization logic.
'''

from __init__ import *


# Library imports.
import imageio
import json
import logging
import wandb


class Logger:
    '''
    Provides generic logging and visualization functionality.
    Uses wandb (weights and biases) and pickle.
    '''

    def __init__(self, log_dir, context):
        '''
        :param log_dir (str): Path to logging folder for this experiment.
        :param context (str): Name of this particular logger instance, for example train / test.
        '''
        self.log_dir = log_dir
        self.context = context
        self.log_path = os.path.join(self.log_dir, context + '.log')
        self.aud_dir = os.path.join(self.log_dir, 'audio')
        self.npy_dir = os.path.join(self.log_dir, 'numpy')
        self.pkl_dir = os.path.join(self.log_dir, 'pickle')
        self.txt_dir = os.path.join(self.log_dir, 'text')
        self.vis_dir = os.path.join(self.log_dir, 'visuals')
        os.makedirs(self.aud_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.npy_dir, exist_ok=True)
        os.makedirs(self.pkl_dir, exist_ok=True)
        os.makedirs(self.txt_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )

        # These lists store losses and other values on an epoch-level time scale, per key.
        self.scalar_memory = collections.defaultdict(list)
        self.scalar_memory_hist = dict()
        self.initialized = False

        # This method call initializes some more variables and resets them for every epoch.
        self.epoch_finished(-1)

    def save_args(self, args):
        '''
        Records all parameters with which the script was called for reproducibility purposes.
        '''
        args_path = os.path.join(self.log_dir, 'args_' + self.context + '.txt')
        with open(args_path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def init_wandb(self, project, args, networks, group='debug', name=None):
        '''
        Initializes the online dashboard, incorporating all PyTorch modules.
        '''
        if name is None:
            name = args.name
        wandb.init(project=project, group=group, config=args, name=name)
        if not isinstance(networks, collections.abc.Iterable):
            networks = [networks]
        for net in networks:
            if net is not None:
                wandb.watch(net)
        self.initialized = True

    def debug(self, *args):
        if args == ():
            args = ['']
        logging.debug(*args)

    def info(self, *args):
        if args == ():
            args = ['']
        logging.info(*args)

    def warning(self, *args):
        if args == ():
            args = ['']
        logging.warning(*args)

    def error(self, *args):
        if args == ():
            args = ['']
        logging.error(*args)

    def critical(self, *args):
        if args == ():
            args = ['']
        logging.critical(*args)

    def exception(self, *args):
        if args == ():
            args = ['']
        logging.exception(*args)

    def report_scalar(self, key, value, step=None, remember=False, commit_histogram=False):
        '''
        Logs a single named value associated with a step.
        If commit_histogram, actual logging is deferred until commit_scalars() is called.
        '''
        if not remember:
            if self.initialized:
                wandb.log({key: value}, step=step)
            else:
                self.debug(str(key) + ': ' + str(value))
        else:
            self.scalar_memory[key].append(value)
            self.scalar_memory_hist[key] = commit_histogram

    def commit_scalars(self, keys=None, step=None):
        '''
        Aggregates a bunch of report_scalar() calls for one or more named sets of values and records
        their histograms, i.e. statistical properties.
        '''
        if keys is None:
            keys = list(self.scalar_memory.keys())
        for key in keys:
            if len(self.scalar_memory[key]) == 0:
                continue

            value = np.mean(self.scalar_memory[key])
            if self.initialized:
                if self.scalar_memory_hist[key]:
                    wandb.log({key: wandb.Histogram(np.array(self.scalar_memory[key]))}, step=step)
                else:
                    wandb.log({key: value}, step=step)

            else:
                self.debug(str(key) + ': ' + str(value))
            self.scalar_memory[key].clear()

    def report_histogram(self, key, value, step=None):
        '''
        Directly logs the statistical properties of a named iterable value, such as a list of
        numbers.
        '''
        if self.initialized:
            wandb.log({key: wandb.Histogram(value)}, step=step)

    def save_image(self, image, step=None, file_name=None, online_name=None, caption=None,
                   upscale_factor=2, accumulate_online=16):
        '''
        Records a single image to a file in visuals and/or the online dashboard.
        '''
        if image.dtype == np.float32:
            image = (image * 255.0).astype(np.uint8)

        if upscale_factor > 1:
            image = cv2.resize(
                image,
                (image.shape[1] * upscale_factor, image.shape[0] * upscale_factor),
                interpolation=cv2.INTER_NEAREST)

        if file_name is not None:
            plt.imsave(os.path.join(self.vis_dir, file_name), image)

        if online_name is not None and self.initialized:
            self.accum_image_buffer[online_name].append(wandb.Image(image, caption=caption))
            if len(self.accum_image_buffer[online_name]) >= accumulate_online:
                wandb.log({online_name: self.accum_image_buffer[online_name]}, step=step)
                self.accum_image_buffer[online_name] = []

    def save_audio(self, waveform, sample_rate=48000, step=None, file_name=None, online_name=None):
        '''
        Records a single audio waveform to a file in audio and/or the online dashboard.
        '''
        if file_name is not None:
            scipy.io.wavfile.write(os.path.join(self.aud_dir, file_name), sample_rate, waveform)
        if online_name is not None and self.initialized:
            wandb.log({online_name: wandb.Audio(waveform, sample_rate=sample_rate)},
                      step=step)

    def save_video(self, frames, step=None, file_name=None, online_name=None, caption=None, fps=6,
                   accumulate_online=16):
        '''
        Records a single set of frames as a video to a file in visuals and/or the online dashboard.
        '''
        # Duplicate last frame for better visibility.
        last_frame = frames[len(frames) - 1:len(frames)]
        frames = np.concatenate([frames, last_frame], axis=0)
        if frames.dtype == np.float32:
            frames = (frames * 255.0).astype(np.uint8)

        if file_name is not None:
            file_path = os.path.join(self.vis_dir, file_name)
            imageio.mimwrite(file_path, frames, fps=fps)

        if online_name is not None and self.initialized:
            assert file_name is not None
            self.accum_video_buffer[online_name].append(
                wandb.Video(file_path, caption=caption, fps=fps, format='gif'))
            if len(self.accum_video_buffer[online_name]) >= accumulate_online:
                wandb.log({online_name: self.accum_video_buffer[online_name]}, step=step)
                self.accum_video_buffer[online_name] = []

    def save_gallery(self, frames, step=None, file_name=None, online_name=None, caption=None,
                     upscale_factor=1, accumulate_online=16):
        '''
        Records a single set of frames as a gallery image to a file in visuals and/or the online
        dashboard.
        '''
        if frames.shape[-1] > 3:  # Grayscale: (..., H, W).
            arrangement = frames.shape[:-2]
        else:  # RGB: (..., H, W, 1/3).
            arrangement = frames.shape[:-3]
        if len(arrangement) == 1:  # (A, H, W, 1/3?).
            gallery = np.concatenate(frames, axis=1)  # (H, A*W, 1/3?).
        elif len(arrangement) == 2:  # (A, B, H, W, 1/3?).
            gallery = np.concatenate(frames, axis=1)  # (B, A*H, W, 1/3?).
            gallery = np.concatenate(gallery, axis=1)  # (A*H, B*W, 1/3?).
        else:
            raise ValueError('Too many dimensions to create a gallery.')
        if gallery.dtype == np.float32 or gallery.dtype == np.float64:
            gallery = (gallery * 255.0).astype(np.uint8)

        if upscale_factor > 1:
            gallery = cv2.resize(
                gallery,
                (gallery.shape[1] * upscale_factor, gallery.shape[0] * upscale_factor),
                interpolation=cv2.INTER_NEAREST)

        self.save_image(gallery, step=step, file_name=file_name, online_name=online_name,
                        caption=caption, upscale_factor=upscale_factor,
                        accumulate_online=accumulate_online)

    def save_numpy(self, array, file_name, step=None, folder=None):
        '''
        Stores a numpy object locally, either in pickle or a chosen directory.
        '''
        if folder is None:
            dst_dp = self.npy_dir
        else:
            dst_dp = os.path.join(self.log_dir, folder)
            os.makedirs(dst_dp, exist_ok=True)
        np.save(os.path.join(dst_dp, file_name), array)

    def save_pickle(self, obj, file_name, step=None, folder=None):
        '''
        Stores a pickle object locally, either in pickle or a chosen directory.
        '''
        if folder is None:
            dst_dp = self.pkl_dir
        else:
            dst_dp = os.path.join(self.log_dir, folder)
            os.makedirs(dst_dp, exist_ok=True)
        dst_fp = os.path.join(dst_dp, file_name)
        with open(dst_fp, 'wb') as f:
            pickle.dump(obj, f)

    def save_text(self, text_obj, file_name, step=None, folder=None):
        '''
        Stores a raw string, or a dict, or a JSON object, as text locally, either in text or a
        chosen directory.
        '''
        if folder is None:
            dst_dp = self.txt_dir
        else:
            dst_dp = os.path.join(self.log_dir, folder)
            os.makedirs(dst_dp, exist_ok=True)
        dst_fp = os.path.join(dst_dp, file_name)

        # https://stackoverflow.com/questions/64154850/convert-dictionary-to-a-json-in-python
        class NumpyFloatValuesEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.float32):
                    return float(obj)
                elif torch.is_tensor(obj):
                    if obj.ndim == 0:
                        return float(obj.item())
                    else:
                        return obj.tolist()  # Handles an arbitrary number of dimensions.
                else:
                    return json.JSONEncoder.default(self, obj)

        if isinstance(text_obj, str):
            with open(dst_fp, 'w') as f:
                f.write(text_obj)
        else:  # Assume dict or json.
            with open(dst_fp, 'w') as f:
                json.dump(text_obj, f, indent=2, cls=NumpyFloatValuesEncoder)

    def epoch_finished(self, epoch):
        # These lists store visuals on a step-level time scale, per key.
        # We clear them every epoch to avoid leaking information between epochs.
        # NOTE: Gallery calls image, so the same buffer dictionary is used.
        self.accum_image_buffer = collections.defaultdict(list)
        self.accum_video_buffer = collections.defaultdict(list)
