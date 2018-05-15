# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:51:03 2017

@author: yuhchen
"""

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg

import pdb

class LabelResizeLayer(caffe.Layer):
    """
    Resize label to be the same size with the samples
    """
    def setup(self, bottom, top):
        feats = bottom[0].data
        lbs = bottom[1].data

        resized_lbs = np.ones((feats.shape[0], 1),dtype=np.float32)
        resized_lbs[:] = lbs[0]
        
        top[0].reshape(*resized_lbs.shape)
        
    def forward(self, bottom, top):
        feats = bottom[0].data
        lbs = bottom[1].data

        resized_lbs = np.ones((feats.shape[0], 1),dtype=np.float32)      
        resized_lbs[:] = lbs[0]
        
        top[0].reshape(*resized_lbs.shape)
        top[0].data[...] = resized_lbs
        
    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
