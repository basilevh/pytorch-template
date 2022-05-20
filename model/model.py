'''
Neural network architecture description.
'''

from __init__ import *

# Internal imports.
import my_utils
import perceiver
import vision_tf


class MySimpleModel(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger):
        '''
        X
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


class MyDenseVitModel(torch.nn.Module):

    def __init__(self, logger):
        super().__init__()

        self.backbone = vision_tf.MyDenseVisionTransformerBackbone(
            logger, 224, 288, 3)

    def forward(self, x):

        y = self.backbone(x)

        return y


class MyPerceiverModel(torch.nn.Module):

    def __init__(self, logger):
        super().__init__()

        self.backbone = perceiver.MyPerceiverBackbone(
            logger, (224, 288), 3, (224, 288), 3, 0, 'fourier')

    def forward(self, x):

        y = self.backbone(x)

        return y
