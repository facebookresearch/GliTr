#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Adapted from: https://github.com/mengyuest/AR-Net/blob/master/ops/dataset.py
"""

import torch.utils.data as data
import torch
from torch.nn import functional as F

from PIL import Image
import os
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row, num_class):
        self._data = row
        self.num_class = num_class
        labels = torch.LongTensor(sorted(list(set([int(x) for x in self._data[2:]]))))
        self._labels = labels

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        labels = F.one_hot(self._labels, num_classes=self.num_class)
        labels = labels.sum(dim=0).bool()
        return labels


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, 
                 dataset=None,
                 num_class=0,
                 ):

        self.root_path     = root_path
        self.num_class     = num_class

        self.list_file = \
            ".".join(list_file.split(".")[:-1]) + "." + list_file.split(".")[-1]  # TODO
        self.num_segments   = num_segments
        self.image_tmpl     = image_tmpl
        self.transform      = transform
        self.random_shift   = random_shift
        self.test_mode      = test_mode
        self.remove_missing = remove_missing

        self.dataset = dataset

        self._parse_list()

    def _load_image(self, directory, idx):
        try:
            img = [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            return img
        except Exception:
            print('couldnt find the data')


    def _parse_list(self):
        # check the frame number is large >3:
        splitter = " "

        with open(self.list_file) as metafile:
            tmp = [x.strip().split(splitter) for x in metafile]

        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]

        self.video_list = [VideoRecord(item, self.num_class) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            print("I am in dataset.py too")
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = record.num_frames // self.num_segments
        if average_duration > 0:
            '''
            divide the video into self.num_segments segments.
                i.e. if video contains 300 frames and self.num_segments=3, each segment is of length(=ticks)=100 frames
            offsets gets a random frame from each segment.
            '''
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames, size=self.num_segments))
        else:
            offsets = np.array(list(range(record.num_frames)) + [record.num_frames - 1] * (self.num_segments - record.num_frames))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments:
            '''
            divide the video into self.num_segments segments.
                i.e. if video contains 300 frames and self.num_segments=3, each segment is of length(=ticks)=100 frames
            offsets gets the center of each segment.
            '''
            tick = record.num_frames / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.array(
                list(range(record.num_frames)) + [record.num_frames - 1] * (self.num_segments - record.num_frames))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = record.num_frames / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            images.extend(self._load_image(record.path, int(seg_ind)))
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)



