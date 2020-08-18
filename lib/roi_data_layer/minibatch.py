# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import copy
import cv2
import pdb
def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  gt_boxes_sens = np.empty((len(gt_inds), 4), dtype=np.float32)
  gt_boxes_sens[:, 0:4] = roidb[0]['boxes_sens'][gt_inds, :] * im_scales[0]
  blobs['gt_boxes'] = gt_boxes
  blobs['gt_boxes_sens'] = gt_boxes_sens
  blobs['im_info'] = np.array(
    [[im_blob[0].shape[1], im_blob[0].shape[2], im_scales[0]]],
    dtype=np.float32)

  blobs['img_id'] = roidb[0]['img_id']

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  assert isinstance(cfg.SHIFT_X, int) and isinstance(cfg.SHIFT_X, int), \
         'wrong shift number, please check'
  for i in range(num_images):
    im = []
    # the reference and sensed modality
    for j in range(2):
      im.append(imread(roidb[i]['image'][j]))
      if len(im[j].shape) == 2: 
        im[j] = im[j][:,:,np.newaxis]
        im[j] = np.concatenate((im[j],im[j],im[j]), axis=2)
      # flip the channel, since the original one using cv2
      # rgb -> bgr
      im[j] = im[j][:,:,::-1]

      if j==1 and (cfg.SHIFT_X!=0 or cfg.SHIFT_Y!=0):
        new_img = np.zeros(im[j].shape)
        if cfg.SHIFT_X>0:
          if cfg.SHIFT_Y>0:
            new_img[:-cfg.SHIFT_Y,cfg.SHIFT_X:,:] = im[j][cfg.SHIFT_Y:,:-cfg.SHIFT_X,:]
          elif cfg.SHIFT_Y<0:
            new_img[-cfg.SHIFT_Y:,cfg.SHIFT_X:,:] = im[j][:cfg.SHIFT_Y,:-cfg.SHIFT_X,:]
          else:
            new_img[:,cfg.SHIFT_X:,:] = im[j][:,:-cfg.SHIFT_X,:]
        elif cfg.SHIFT_X<0:
          if cfg.SHIFT_Y>0:
            new_img[:-cfg.SHIFT_Y,:cfg.SHIFT_X,:] = im[j][cfg.SHIFT_Y:,-cfg.SHIFT_X:,:]
          elif cfg.SHIFT_Y<0:
            new_img[-cfg.SHIFT_Y:,:cfg.SHIFT_X,:] = im[j][:cfg.SHIFT_Y,-cfg.SHIFT_X:,:]
          else:
            new_img[:,:cfg.SHIFT_X,:] = im[j][:,-cfg.SHIFT_X:,:]
        else:
          if cfg.SHIFT_Y>0:
            new_img[:-cfg.SHIFT_Y,:,:] = im[j][cfg.SHIFT_Y:,:,:]
          elif cfg.SHIFT_Y<0:
            new_img[-cfg.SHIFT_Y:,:,:] = im[j][:cfg.SHIFT_Y,:,:] 
          else:
            pass
        im[j] = new_img

      if roidb[i]['flipped']:
        im[j] = im[j][:, ::-1, :]
      target_size = cfg.TRAIN.SCALES[scale_inds[i]]
      im[j], im_scale = prep_im_for_blob(im[j], cfg.PIXEL_MEANS, target_size,
                      cfg.TRAIN.MAX_SIZE)

    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
