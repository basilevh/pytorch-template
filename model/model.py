'''
Neural network architecture description.
'''

from __init__ import *

# Internal imports.
import backbone
import utils


class MySimpleModel(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger):
        '''
        :param image_dim (int): Size of entire input or output image.
        :param patch_dim (int): Size of one image patch.
        :param emb_dim (int): Internal feature embedding size.
        :param depth (int): Perceiver IO depth.
        '''
        super().__init__()
        self.logger = logger
        self.net = torch.nn.Conv2d(3, 3, 1)

    def forward(self, rgb_input):
        '''
        :param rgb_input (B, 3, Hi, Wi) tensor.
        :return rgb_output (B, 3, Hi, Wi) tensor.
        '''
        rgb_output = self.net(rgb_input)
        return rgb_output


class MyImageToAudioModel(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger, image_height, image_width, num_channels,
                 audio_samples, samples_per_frame, output_pos_enc):
        '''
        :param image_height, image_width, num_channels (int): Input image dimensions.
        :param audio_samples (int): Output waveform length.
        '''
        super().__init__()
        self.logger = logger
        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = num_channels
        self.audio_samples = audio_samples

        self.model = backbone.MyPerceiverBackbone(
            logger, (image_height, image_width), (audio_samples, ), samples_per_frame,
            output_pos_enc)

    def forward(self, control_matrix):
        '''
        :param control_matrix (B, C, T) tensor.
        :return (waveform, last_hidden_state).
            waveform (B, S) tensor.
            last_hidden_state (B, L, D) tensor.
        '''

        (waveform, last_hidden_state) = self.model(control_matrix)

        return (waveform, last_hidden_state)
