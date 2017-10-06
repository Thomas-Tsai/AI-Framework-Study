import os, sys, caffe
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold='nan')

modelF = '/opt/caffe/examples/mnist/lenet.prototxt'
#trainedF = '/opt/caffe/examples/mnist/lenet_iter_10000.caffemodel'
trainedF = '/opt/caffe/examples/mnist/lenet_iter_5000.caffemodel'
imgF = "/opt/caffe/thomas/train_8.bmp"

net = caffe.Net(modelF, trainedF, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)
net.blobs['data'].reshape(1, 1, 28, 28)

img = caffe.io.load_image(imgF, color=False)

print "Show the size of image:\n",img.shape,"\n"

net.blobs['data'].data[...] = transformer.preprocess('data', img)

out = net.forward()
print "The type of the data come from net is Blob:\n",out,"\n"
print "We need to convert it to matrix that we can use:\n",out[net.outputs[0]],"\n"
print "This is second method that can extract any data from any layer:\n",net.blobs['prob'].data,"\n"

