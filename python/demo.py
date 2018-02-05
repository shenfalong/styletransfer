#!/usr/bin/env python

import _pycaffe as caffe
from cv2 import *
import pdb
import numpy as np
import time

def pycaffe_hidden(im_label):
	prototxt_file = 'prototxt/8/hidden.prototxt'
	weights_file = 'model/train_8.caffemodel'
	if caffe.is_initialized() < 1:
		caffe.init(prototxt_file, weights_file)
		caffe.set_device(0)
	
	im_label = im_label.astype('float32')

	im_label[:,:,0] = im_label[:,:,0] - 104.008
	im_label[:,:,1] = im_label[:,:,1] - 116.669
	im_label[:,:,2] = im_label[:,:,2] - 122.675
	im_label_ = np.expand_dims(im_label,3)
	im_label = np.transpose(im_label_,(3,2,0,1)).copy()


	input_data = [im_label]
	score = caffe.forward(0,input_data)
	hidden_feat = score[0].squeeze()
	
	return hidden_feat
	
def pycaffe_param(hidden_feat):
	prototxt_file = 'prototxt/8/param.prototxt'
	weights_file = 'model/train_8.caffemodel'
	if caffe.is_initialized() < 2:
		caffe.init(prototxt_file, weights_file)
		caffe.set_device(0)
	
	hidden_feat = hidden_feat.reshape((1,hidden_feat.size,1,1))
	input_data = [hidden_feat]
	param = caffe.forward(1, input_data)
	
	
	caffe.save_model(param,'layer_name.txt','base.caffemodel','predict.caffemodel')

def pycaffe_predict(im):
	prototxt_file = 'prototxt/8/predict.prototxt'
	weights_file = 'predict.caffemodel'
	if caffe.is_initialized() < 3:
		caffe.init(prototxt_file, weights_file)
		caffe.set_device(0)
		
	im = im.astype('float32')
	im[:,:,0] = im[:,:,0] - 104.008
	im[:,:,1] = im[:,:,1] - 116.669
	im[:,:,2] = im[:,:,2] - 122.675
	im = np.expand_dims(im,3)
	im = np.transpose(im,(3,2,0,1)).copy()
	
	input_data = [im]
	t1=time.time()
	score = caffe.forward(2, input_data)
	t2=time.time()
	print t2-t1
	
	
	
	raw_score = score[0]
	
	
	raw_score = raw_score[0,:,:,:]
	
	raw_score = np.transpose(raw_score,(1,2,0)).copy()
	
	
	raw_score[:,:,0] = raw_score[:,:,0] + 104.008
	raw_score[:,:,1] = raw_score[:,:,1] + 116.669
	raw_score[:,:,2] = raw_score[:,:,2] + 122.675
	
	raw_score = np.clip(raw_score,0,255)
	
	return raw_score.astype('uint8')
	
if __name__ == '__main__':
	caffe.base_model('model/train_8.caffemodel', 'base.txt')
	

	style_im = imread('jianbihua.png')
	#style_im = resize(style_im,(256,256))
	hidden_feat = pycaffe_hidden(style_im)
	

	pycaffe_param(hidden_feat)
	
	origin_im = imread('content.jpg')
	scoremap = pycaffe_predict(origin_im)
	imwrite('result.png',scoremap)

