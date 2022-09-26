'''
Neural network architecture description.
'''

from __init__ import *

# Internal imports.
import my_utils
# import perceiver
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
