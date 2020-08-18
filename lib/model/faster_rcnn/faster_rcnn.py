import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.init as init
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import copy
import pdb
from model.rpn.bbox_transform import bbox_contextual_batch, clip_boxes, bbox_transform_inv
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/8.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/8.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # CAF head
        self.confidence_ref = nn.Linear(4096, 2)
        self.confidence_sens = nn.Linear(4096, 2)

    def forward(self, im_data, im_info, gt_boxes, gt_boxes_sens, num_boxes):
        batch_size = im_data[0].size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        gt_boxes_sens = gt_boxes_sens.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat_c = self.RCNN_base_c(im_data[0])
        base_feat_t = self.RCNN_base_t(im_data[1])
        base_feat_fused = 0.5 * (base_feat_c + base_feat_t)
        base_feat_fused = self.RCNN_base_fused(base_feat_fused)
        conv5_c = self.RCNN_base_f1(base_feat_c)
        conv5_t = self.RCNN_base_f2(base_feat_t)

        # feed fused base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_fused, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            # 50% jitter probability
            if np.random.rand(1)[0]>0.5:
                jitter = (torch.randn(1,256,4)/20).cuda()
            else:
                jitter = (torch.zeros(1,256,4)).cuda()
            # feed jitter to obtain rois_align_target
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, gt_boxes_sens, num_boxes, jitter, im_info)
            rois, rois_jittered, rois_label, rois_target, rois_align_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_align_target = Variable(rois_align_target.view(-1, rois_align_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_jittered = copy.deepcopy(rois)
            rois_label = None
            rois_target = None
            rois_align_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0


        # Region Feature Alignment module
        ctx_rois = bbox_contextual_batch(rois)
        clip_boxes(ctx_rois[:,:,1:], im_info, batch_size)
        ctx_rois = Variable(ctx_rois)
        ctx_rois_jittered = bbox_contextual_batch(rois_jittered)
        clip_boxes(ctx_rois_jittered[:,:,1:], im_info, batch_size)
        ctx_rois_jittered = Variable(ctx_rois_jittered)

        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(ctx_rois.view(-1, 5), conv5_c.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat_c = self.RCNN_roi_crop(conv5_c, Variable(grid_yx).detach())
            grid_xy = _affine_grid_gen(ctx_rois_jittered.view(-1, 5), conv5_t.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat_t = self.RCNN_roi_crop(conv5_t, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_c = F.max_pool2d(pooled_feat_c, 2, 2)
                pooled_feat_t = F.max_pool2d(pooled_feat_t, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat_c = self.RCNN_roi_align(conv5_c, ctx_rois.view(-1, 5))    
            pooled_feat_t = self.RCNN_roi_align(conv5_t, ctx_rois_jittered.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_c = self.RCNN_roi_pool(conv5_c, ctx_rois.view(-1,5))
            pooled_feat_t = self.RCNN_roi_pool(conv5_t, ctx_rois_jittered.view(-1,5))
        
        pooled_feat_res = pooled_feat_t - pooled_feat_c

        # feed pooled features to top model
        pooled_feat_res = self._head_to_tail_align(pooled_feat_res)
        bbox_align_pred = self.RCNN_bbox_align_pred(pooled_feat_res)

        RCNN_loss_bbox_align = 0
        
        # Apply bounding-box regression deltas
        box_deltas = bbox_align_pred.data
        box_deltas_zeros = torch.zeros(box_deltas.shape).cuda()
        box_deltas = torch.cat((box_deltas, box_deltas_zeros), 1)


        # Optionally normalize targets by a precomputed mean and stdev
        # The roi alignment process is class_agnostic
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(batch_size, -1, 4)

        rois_sens = rois_jittered.new(rois_jittered.size()).zero_()
        rois_sens[:,:,1:5] = bbox_transform_inv(rois_jittered[:,:,1:5], box_deltas, batch_size)

        clip_boxes(rois_sens[:,:,1:5], im_info, batch_size)
        


        rois = Variable(rois)
        rois_sens = Variable(rois_sens)

        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), conv5_c.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat_c = self.RCNN_roi_crop(conv5_c, Variable(grid_yx).detach())
            grid_xy = _affine_grid_gen(rois_sens.view(-1, 5), conv5_t.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat_t = self.RCNN_roi_crop(conv5_t, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_c = F.max_pool2d(pooled_feat_c, 2, 2)
                pooled_feat_t = F.max_pool2d(pooled_feat_t, 2, 2)

        elif cfg.POOLING_MODE == 'align':
            pooled_feat_c = self.RCNN_roi_align(conv5_c, rois.view(-1, 5))
            pooled_feat_t = self.RCNN_roi_align(conv5_t, rois_sens.view(-1, 5))

        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_c = self.RCNN_roi_pool(conv5_c, rois.view(-1, 5))
            pooled_feat_t = self.RCNN_roi_pool(conv5_t, rois_sens.view(-1, 5))
                                                        
        cls_score_ref = self.confidence_ref(self.RCNN_top_ref(pooled_feat_c.view(pooled_feat_c.size(0), -1)))
        cls_score_sens = self.confidence_sens(self.RCNN_top_sens(pooled_feat_t.view(pooled_feat_t.size(0), -1)))
        cls_prob_ref = F.softmax(cls_score_ref, 1)
        cls_prob_sens = F.softmax(cls_score_sens, 1)

        confidence_ref = torch.abs(cls_prob_ref[:,1]-cls_prob_ref[:,0])
        confidence_sens = torch.abs(cls_prob_sens[:,1]-cls_prob_sens[:,0])
        confidence_ref = confidence_ref.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        confidence_sens = confidence_sens.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        pooled_feat_c = confidence_ref * pooled_feat_c
        pooled_feat_t = confidence_sens * pooled_feat_t
        pooled_feat = pooled_feat_c + pooled_feat_t


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_cls_ref = 0
        RCNN_loss_cls_sens = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_cls_ref = F.cross_entropy(cls_score_ref, rois_label)
            RCNN_loss_cls_sens = F.cross_entropy(cls_score_sens, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            RCNN_loss_bbox_align = _smooth_l1_loss(bbox_align_pred, rois_align_target[:,:2], rois_inside_ws[:,:2], rois_outside_ws[:,:2])


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, rois_sens, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_cls_ref, RCNN_loss_cls_sens, RCNN_loss_bbox, RCNN_loss_bbox_align, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def xavier(param):
          init.xavier_uniform(param)
        def xavier_init(m):
          if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_align_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)


    def create_architecture(self):
        self._init_modules()
        self._init_weights()
