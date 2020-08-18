# -*- coding:utf-8 -*- 
from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
import pdb

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete

class kaist(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'kaist_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = self._devkit_path
        self._classes = ('__background__',  # always index 0
                         'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2,
                       'person_only': False,
                       'ref_sens_mods': ['lwir', 'visible']} 

        assert os.path.exists(self._devkit_path), \
            'KAISTdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        indexes = index.split('/')
        image_paths = []
        for mod in self.config['ref_sens_mods']: 
            path = os.path.join(self._data_path, 'images', 
                                  indexes[0], indexes[1], mod,   
                                  indexes[2] + self._image_ext)
            assert os.path.exists(path), \
                'Path does not exist: {}'.format(path)
            image_paths.append(path)

        return image_paths


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /splits/trainval.txt
        image_set_file = os.path.join(self._data_path, 'splits',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where KAIST-Paired is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'kaist-paired')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_kaist_paired_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_kaist_paired_annotation(self, index):
        """
        Load image and bounding boxes info from TXT file.
        """
        
        # Load annotations of the reference and sensed modality
        for idx, mod in enumerate(self.config['ref_sens_mods']):
            indexes = index.split('/')
            filename = os.path.join(self._data_path, 'annotations', indexes[0], indexes[1], \
                                        mod, indexes[2] + '.txt') 
            with open(filename, 'r') as f:
                objs = f.readlines()[1:]
            if self.config['person_only']:
                person_only_objs = [obj for obj in objs if obj.split(' ')[0].lower().strip() == 'person']
                if len(person_only_objs) != len(objs):
                    print('Removed {} non-person reference objects'.format(
                        len(objs) - len(person_only_objs)))
                objs = person_only_objs
            else:
                remain_objs = []
                for obj in objs:
                    obj_split = obj.split(' ')
                    if obj_split[0].lower().strip() == 'person' and obj_split[5].strip() != '2':
                        remain_objs.append(obj)
                    # cyclist as positive sample
                    elif obj_split[0].lower().strip() == 'cyclist':
                        obj_split[0] = 'person'
                        remain_objs.append(' '.join(obj_split))
                objs = remain_objs
                
            num_objs = len(objs)

            if idx == 0:
                boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_objs), dtype=np.int32)
                overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
                # "Seg" area for pascal is just the box area
                seg_areas = np.zeros((num_objs), dtype=np.float32)
                ishards = np.zeros((num_objs), dtype=np.int32)
            else:
                boxes_sens = np.zeros((num_objs, 4), dtype=np.uint16)

            for ix, obj in enumerate(objs):
                obj = obj.split(' ')
                cls = self._class_to_ind[obj[0].lower().strip()]
                # Make pixel indexes 0-based
                x1 = float(obj[1]) - 1
                y1 = float(obj[2]) - 1
                x2 = x1 + float(obj[3])
                y2 = y1 + float(obj[4])

                if idx == 0:
                    #diffc = obj.find('difficult')
                    difficult = 0 # if diffc == None else int(diffc.text)
                    ishards[ix] = difficult

                    boxes[ix, :] = [x1, y1, x2, y2]
                    gt_classes[ix] = cls
                    overlaps[ix, cls] = 1.0
                    seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                else:
                    boxes_sens[ix, :] = [x1, y1, x2, y2]
            
            if idx == 0:
                overlaps = scipy.sparse.csr_matrix(overlaps)

        assert boxes.shape == boxes_sens.shape, \
            'Number of objects are mismatched at: {}'.format(index)


        return {'boxes': boxes,
                'boxes_sens': boxes_sens,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id


    def _do_matlab_eval(self, output_dir='output'):
        pass

    def evaluate_detections(self, all_boxes, output_dir):
        pass

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = kaist('trainval')
    res = d.roidb
    from IPython import embed;

    embed()
