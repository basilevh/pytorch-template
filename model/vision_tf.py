'''
Neural network architectures for dense prediction via transformer-based image and/or video models.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'model/'))
sys.path.insert(0, os.getcwd())

from __init__ import *

# Library imports.
import os
import sys
import timm

# Internal imports.
from timesformer.models.vit import TimeSformer


# NOTE: Not used in my augs, BUT used in most pretrained models.
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/constants.py
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/config/defaults.py
TIMESFORMER_MEAN = (0.45, 0.45, 0.45)
TIMESFORMER_STD = (0.225, 0.225, 0.225)


class DenseVisionTransformer(torch.nn.Module):
    '''
    Based on https://github.com/rwightman/pytorch-image-models.
    '''

    def __init__(self, logger, timm_name, pretrained_frozen, frame_height, frame_width, patch_dim,
                 in_channels):
        super().__init__()
        self.logger = logger
        self.timm_name = timm_name
        self.pretrained = pretrained_frozen
        # Frame size.
        self.Hf = frame_height
        self.Wf = frame_width
        # Number of patches.
        self.Ho = frame_height // patch_dim
        self.Wo = frame_width // patch_dim
        # Patch size.
        self.ho = patch_dim
        self.wo = patch_dim
        # Number of channels.
        self.Ci = in_channels

        # NOTE: We are usually modifying the image size which results in a different token sequence
        # length and set of positional embeddings. Not sure what the effect is.
        self.vit = timm.create_model(timm_name, pretrained=pretrained_frozen,
                                     img_size=(self.Hf, self.Wf))
        assert self.ho == 16 and self.wo == 16
        self.output_feature_dim = 768

        if pretrained_frozen:
            # Disable gradients for target features.
            # NOTE: used_model must always be set to eval, regardless of this model phase.
            for param in self.vit.parameters():
                param.requires_grad_(False)

        # Replace first convolutional layer to accommodate non-standard inputs.
        if in_channels != 3:
            assert not(pretrained_frozen)
            self.vit.patch_embed.proj = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=768, kernel_size=(16, 16), stride=(16, 16))

    def forward(self, input_pixels):
        '''
        :param input_pixels (B, C, H, W) tensor.
        '''

        # Normalize if pretrained.
        if self.pretrained:
            mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=input_pixels.dtype,
                                device=input_pixels.device)
            mean = mean[:, None, None].expand_as(input_pixels[0])
            std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=input_pixels.dtype,
                               device=input_pixels.device)
            std = std[:, None, None].expand_as(input_pixels[0])
            input_pixels = input_pixels - mean
            input_pixels = input_pixels / std

        # Adapted from
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py.
        x = self.vit.patch_embed(input_pixels)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)  # (B, N, D), where N = 1 (cls_token) + Ho * Wo.

        # Discard cls_token altogether, and skip norm, pre_logits, head.
        x = x[:, 1:]  # (B, Ho * Wo, D).

        # Refer to
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py.
        # Here, we undo x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC.
        x = rearrange(x, 'B (H W) D -> B D H W', H=self.Ho, W=self.Wo)
        output_features = x

        assert output_features.shape[1] == self.output_feature_dim

        return output_features


class DenseTimeSformer(torch.nn.Module):
    '''
    Based on https://github.com/facebookresearch/TimeSformer.
    '''

    def __init__(self, logger, pretrained_frozen, pretrained_path, frame_height, frame_width,
                 patch_dim, in_channels, num_frames, attention_type):
        super().__init__()
        self.logger = logger
        self.pretrained = pretrained_frozen
        # Frame size.
        self.Hf = frame_height
        self.Wf = frame_width
        # Number of patches.
        self.Ho = frame_height // patch_dim
        self.Wo = frame_width // patch_dim
        # Patch size.
        self.ho = patch_dim
        self.wo = patch_dim
        # Number of channels.
        self.Ci = in_channels
        # Extra options.
        self.T = num_frames
        self.attention_type = attention_type

        self.timesformer = TimeSformer(
            img_size=(self.Hf, self.Wf), patch_size=16, num_classes=0, num_frames=self.T,
            attention_type=self.attention_type, in_chans=self.Ci,
            pretrained_model=pretrained_path)
        assert self.ho == 16 and self.wo == 16
        self.output_feature_dim = 768

        if pretrained_frozen:
            # Disable gradients for target features.
            # NOTE: used_model must always be set to eval, regardless of this model phase.
            for param in self.timesformer.parameters():
                param.requires_grad_(False)

        # Taken from their dataset code (Kinetics and SSv2):
        self.data_mean = [0.45, 0.45, 0.45]
        self.data_std = [0.225, 0.225, 0.225]

    def forward(self, input_pixels):
        '''
        :param input_pixels (B, C, T, H, W) tensor.
        '''

        # Normalize if pretrained.
        if self.pretrained:
            mean = torch.tensor(TIMESFORMER_MEAN, dtype=input_pixels.dtype,
                                device=input_pixels.device)
            mean = mean[:, None, None, None].expand_as(input_pixels[0])
            std = torch.tensor(TIMESFORMER_STD, dtype=input_pixels.dtype,
                               device=input_pixels.device)
            std = std[:, None, None, None].expand_as(input_pixels[0])
            input_pixels = input_pixels - mean
            input_pixels = input_pixels / std

        # Adapted from
        # https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py
        # See third_party/timesformer/... for actual code.
        B = input_pixels.shape[0]
        x, T, W = self.timesformer.model.patch_embed(input_pixels)
        assert T == self.T
        assert W == self.Wo

        cls_tokens = self.timesformer.model.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # OLD:
        # assert x.size(1) == self.timesformer.model.pos_embed.size(1)
        # x = x + self.timesformer.model.pos_embed

        # NEW:
        # resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.timesformer.model.pos_embed.size(1):
            pos_embed = self.timesformer.model.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = torch.nn.functional.interpolate(
                other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.timesformer.model.pos_embed

        x = self.timesformer.model.pos_drop(x)

        # Time Embeddings
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)

        # OLD:
        # assert T == self.timesformer.model.time_embed.size(1)
        # x = x + self.timesformer.model.time_embed

        # NEW:
        # Resizing time embeddings in case they don't match
        if T != self.timesformer.model.time_embed.size(1):
            time_embed = self.timesformer.model.time_embed.transpose(1, 2)
            new_time_embed = torch.nn.functional.interpolate(
                time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.timesformer.model.time_embed

        x = self.timesformer.model.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        # Attention blocks
        for blk in self.timesformer.model.blocks:
            x = blk(x, B, T, W)

        # Discard cls_token altogether, and skip norm, head.
        x = x[:, 1:]

        x = rearrange(x, 'B (H W T) D -> B D T H W',
                      B=B, T=T, H=self.Ho, W=self.Wo, D=self.output_feature_dim)
        output_features = x

        assert output_features.shape[1] == self.output_feature_dim

        return output_features


class MyDenseVisionTransformerBackbone(DenseVisionTransformer):
    '''
    Trainable variant of the DenseVisionTransformer.
    '''

    def __init__(self, logger, frame_height=224, frame_width=224, in_channels=3):
        # TODO: Currently hardcoded to vit_base_patch16_224.
        super().__init__(logger, 'vit_base_patch16_224', False, frame_height, frame_width,
                         16, in_channels)


class MyDenseTimeSformerBackbone(DenseTimeSformer):
    '''
    Trainable variant of the DenseTimeSformerBackbone.
    '''

    def __init__(self, logger, num_frames=16, frame_height=224, frame_width=224,
                 in_channels=3, attention_type='divided_space_time'):
        super().__init__(logger, False, '', frame_height, frame_width, 16, in_channels, num_frames,
                         attention_type)


if __name__ == '__main__':

    (B, T, H, W, C) = (2, 18, 192, 160, 3)
    
    print('MyDenseVisionTransformerBackbone')
    my_vit = MyDenseVisionTransformerBackbone(None, H, W, C)
    
    x = torch.randn(B, C, H, W)
    print('x:', x.shape, x.min().item(), x.mean().item(), x.max().item())
    
    y = my_vit(x)
    print('y:', y.shape, y.min().item(), y.mean().item(), y.max().item())
    print()
    
    assert y.shape == (B, my_vit.output_feature_dim, H // 16, W // 16)
    
    for attention_type in ['divided_space_time', 'joint_space_time']:
        
        print('MyDenseTimeSformerBackbone')
        print('attention_type:', attention_type)
        my_tsf = MyDenseTimeSformerBackbone(None, T, H, W, C, attention_type)
        
        x = torch.randn(B, C, T, H, W)
        print('x:', x.shape, x.min().item(), x.mean().item(), x.max().item())
        
        y = my_tsf(x)
        print('y:', y.shape, y.min().item(), y.mean().item(), y.max().item())
        print()
        
        assert y.shape == (B, my_tsf.output_feature_dim, T, H // 16, W // 16)

    pass
