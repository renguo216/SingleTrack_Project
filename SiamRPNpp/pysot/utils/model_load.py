# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict, require_match=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    if require_match:
        assert len(used_pretrained_keys) > 0, \
            'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    # 尝试不同的前缀组合来找匹配
    prefixes_to_try = [
        '',                         # 无前缀
        ('backbone.', ''),          # 移除backbone前缀
        ('features.', ''),          # 移除features前缀
        ('backbone.', 'backbone.'), # 保持不变
    ]
    
    success = False
    for prefix_info in prefixes_to_try:
        if prefix_info == '':
            # 无需修改
            current_dict = pretrained_dict.copy()
            prefix_desc = '(no change)'
        else:
            old_prefix, new_prefix = prefix_info
            current_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith(old_prefix):
                    k = new_prefix + k[len(old_prefix):]
                current_dict[k] = v
            prefix_desc = '{} -> {}'.format(repr(old_prefix), repr(new_prefix))
        
        try:
            check_keys(model, current_dict, require_match=True)
            pretrained_dict = current_dict
            logger.info('[Success] prefix {} matched!'.format(prefix_desc))
            success = True
            break
        except:
            pass
    
    if not success:
        logger.info('[Warning]: No prefix combination matched, training from scratch')
        check_keys(model, pretrained_dict, require_match=False)

    model_state_dict = model.state_dict()
    for k, v in pretrained_dict.items():
        if k in model_state_dict and v.shape != model_state_dict[k].shape:
            if 'downsample.0.weight' in k and v.dim() == 4:
                logger.info('[Warning]: resize {} from {} to {}'.format(
                    k, v.shape, model_state_dict[k].shape))
                new_v = torch.zeros(model_state_dict[k].shape, dtype=v.dtype, device=v.device)
                new_v[:, :, 1:2, 1:2] = v
                pretrained_dict[k] = new_v

    model.load_state_dict(pretrained_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']

    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch
