## Deep Learning Frameworks

使用不同的框架來實作CNN訓練

* Caffe
* Theano
* Torch

## Caffe
* CIFAR-10
* 60000 32X32 color images
* split into 10 classes
  * airplane, automobile, bird, cat, deer
  * dog, frog, horse, ship, truck
* $caffe_path/data/get_cifar10.sh
* $caffe_path/examples/cifar10/create_cifar10.sh
* Blob，Layer，Net，Solve

## Caffe - Blob --Thomas補充

~~~c
The type of the data come from net is Blob:
{'prob': array([[ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.]], dtype=float32)} 

We need to convert it to matrix that we can use:
[[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]] 

This is second method that can extract any data from any layer:
[[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]] 
~~~

## Caffe - Layer --Thomas補充

~~~c
layer {
  name: "conv1"
  type: "Convolution" 
// type: Data, Convolution, Pooling, InnerProduct, ReLU, LRN, 
//       DropOut, SoftmaxWithLoss, Accuracy
// type = Data: data (lmdb, leveldb), MemoryData
//       HDF5Data, ImagesData, WindowsData, DummyData

  bottom: "data" # 輸入層：數據層
  top: "conv1" # 輸出層：卷積層1

  # 濾波器（filters）的學習速率因子和衰減因子
  param { lr_mult: 1 decay_mult: 1 }

  # 偏置項（biases）的學習速率因子和衰減因子
  param { lr_mult: 2 decay_mult: 0 }

  convolution_param {
    num_output: 96 # 96個濾波器（filters）
    kernel_size: 11 # 每個濾波器（filters）大小為11*11
    stride: 4 # 每次濾波間隔為4個像素
    weight_filler {
      type: "gaussian" # 初始化高斯濾波器（Gaussian）
      std: 0.01 # 標準差為0.01， 均值默認為0
    }
    bias_filler {
      type: "constant" # 初始化偏置項（bias）為零
      value: 0
    }
  }
//source code: src/layers/convolution_layer.cpp | cu
}
~~~


## Caffe - net

~~~c 
...
layer {
  name: "cifar"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    mean_file: "examples/cifar10/mean.binaryproto"
  }
  data_param {
    source: "examples/cifar10/cifar10_train_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  ...
  }
}
...
~~~

## Caffe - solver

~~~c
# reduce the learning rate after 8 epochs (4000 iters) by a factor of 10

# The train/test net protocol buffer definition
net: "examples/cifar10/cifar10_quick_train_test.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100
# Carry out testing every 500 training iterations.
test_interval: 500
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.001
momentum: 0.9
weight_decay: 0.004
# The learning rate policy
lr_policy: "fixed"
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 4000
# snapshot intermediate results
snapshot: 4000
snapshot_prefix: "examples/cifar10/cifar10_quick"
# solver mode: CPU or GPU
solver_mode: GPU
~~~

## Caffe - train

~~~bash
cd $caffe_path
./build/tools/caffe train --solver=./examples/cifar10/cifar10_quick_solver.prototxt

// resume training from the half-way point snapshot
// caffe train -solver examples/mnist/lenet_solver.prototxt -snapshot examples/mnist/lenet_iter_5000.solverstate
~~~

## Caffe - time --Thomas

~~~bash
caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10

I1005 05:58:47.061524  1233 caffe.cpp:374] *** Benchmark begins ***
I1005 05:58:47.061544  1233 caffe.cpp:375] Testing for 10 iterations.
I1005 05:58:47.115362  1233 caffe.cpp:403] Iteration: 1 forward-backward time: 53 ms.
I1005 05:58:47.169019  1233 caffe.cpp:403] Iteration: 2 forward-backward time: 53 ms.
I1005 05:58:47.222576  1233 caffe.cpp:403] Iteration: 3 forward-backward time: 53 ms.
I1005 05:58:47.275920  1233 caffe.cpp:403] Iteration: 4 forward-backward time: 53 ms.
I1005 05:58:47.329260  1233 caffe.cpp:403] Iteration: 5 forward-backward time: 53 ms.
I1005 05:58:47.383637  1233 caffe.cpp:403] Iteration: 6 forward-backward time: 54 ms.
I1005 05:58:47.438005  1233 caffe.cpp:403] Iteration: 7 forward-backward time: 54 ms.
I1005 05:58:47.491442  1233 caffe.cpp:403] Iteration: 8 forward-backward time: 53 ms.
I1005 05:58:47.544661  1233 caffe.cpp:403] Iteration: 9 forward-backward time: 53 ms.
I1005 05:58:47.598136  1233 caffe.cpp:403] Iteration: 10 forward-backward time: 53 ms.
I1005 05:58:47.598157  1233 caffe.cpp:406] Average time per layer: 
I1005 05:58:47.598170  1233 caffe.cpp:409]      mnist	forward: 0.0107 ms.
I1005 05:58:47.598176  1233 caffe.cpp:412]      mnist	backward: 0.0006 ms.
I1005 05:58:47.598179  1233 caffe.cpp:409]      conv1	forward: 5.9642 ms.
I1005 05:58:47.598182  1233 caffe.cpp:412]      conv1	backward: 5.9309 ms.
I1005 05:58:47.598186  1233 caffe.cpp:409]      pool1	forward: 2.7406 ms.
I1005 05:58:47.598188  1233 caffe.cpp:412]      pool1	backward: 0.5376 ms.
I1005 05:58:47.598191  1233 caffe.cpp:409]      conv2	forward: 10.0786 ms.
I1005 05:58:47.598196  1233 caffe.cpp:412]      conv2	backward: 19.9588 ms.
I1005 05:58:47.598208  1233 caffe.cpp:409]      pool2	forward: 1.6229 ms.
I1005 05:58:47.598212  1233 caffe.cpp:412]      pool2	backward: 0.686 ms.
I1005 05:58:47.598214  1233 caffe.cpp:409]        ip1	forward: 1.9727 ms.
I1005 05:58:47.598217  1233 caffe.cpp:412]        ip1	backward: 3.7615 ms.
I1005 05:58:47.598220  1233 caffe.cpp:409]      relu1	forward: 0.0241 ms.
I1005 05:58:47.598224  1233 caffe.cpp:412]      relu1	backward: 0.0282 ms.
I1005 05:58:47.598227  1233 caffe.cpp:409]        ip2	forward: 0.1135 ms.
I1005 05:58:47.598230  1233 caffe.cpp:412]        ip2	backward: 0.1435 ms.
I1005 05:58:47.598239  1233 caffe.cpp:409]       loss	forward: 0.0371 ms.
I1005 05:58:47.598243  1233 caffe.cpp:412]       loss	backward: 0.0018 ms.
I1005 05:58:47.598250  1233 caffe.cpp:417] Average Forward pass: 22.5693 ms.
I1005 05:58:47.598254  1233 caffe.cpp:419] Average Backward pass: 31.0536 ms.
I1005 05:58:47.598258  1233 caffe.cpp:421] Average Forward-Backward: 53.6 ms.
I1005 05:58:47.598263  1233 caffe.cpp:423] Total Time: 536 ms.
I1005 05:58:47.598264  1233 caffe.cpp:424] *** Benchmark ends ***
~~~

## Caffe - Python - API

~~~python
# Import required Python libraries 
%matplotlib inline 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import caffe 
import random 
 
# Choose network definition file and pretrained network binary 
MODEL_FILE = '/home/ubuntu/caffe/examples/cifar10/cifar10_quick.prototxt' 
PRETRAINED = '/home/ubuntu/caffe/examples/cifar10/cifar10_quick_iter_4000.caffemodel' 
 
# Load a random image 
x = caffe.io.load_image('/home/ubuntu/caffe/examples/images/' + str(random.randint(1,18)) + '.png') 
 
# Display the chosen image 
plt.imshow(x) 
plt.axis('off') 
plt.show() 
 
# Load the pretrained model and select to use the GPU for computation 
caffe.set_mode_gpu() 
net = caffe.Classifier(MODEL_FILE, PRETRAINED, 
                       mean=np.load('/home/ubuntu/caffe/caffe/examples/cifar10/cifar10_mean.npy').mean(1).mean(1), 
                       raw_scale=255, 
                       image_dims=(32, 32)) 
 
# Run the image through the pretrained network 
prediction = net.predict([x]) 
 
# List of class labels 
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
 
# Display the predicted probability for each class 
plt.plot(prediction[0]) 
plt.xticks(range(0,10), classes, rotation=45) 
# Display the most probable class 
print classes[prediction[0].argmax()] 

~~~

## Theano

* MNIST
* 28X28 black and white images
* handwritten digits
* http://yann.lecun.com/exdb/mnist/

## Theano - Python class for Convolutional Layers with Max Pooling

~~~python
import os 
os.chdir('/home/ubuntu/notebook/DLIntro') 
import sys 
import timeit 
import numpy 
import theano 
import theano.tensor as T 
from theano.tensor.signal import downsample 
from theano.tensor.nnet import conv 
from logistic_sgd import LogisticRegression, load_data 
from mlp import HiddenLayer 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
from utils import tile_raster_images 
from IPython import display 
 
class LeNetConvPoolLayer(object): 
    """Pool Layer of a convolutional network """ 
 
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)): 
 
        assert image_shape[1] == filter_shape[1] 
        self.input = input 
 
        # there are "num input feature maps * filter height * filter width" 
        # inputs to each hidden unit 
        fan_in = numpy.prod(filter_shape[1:]) 
        # each unit in the lower layer receives a gradient from: 
        # "num output feature maps * filter height * filter width" / 
        #   pooling size 
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / 
                   numpy.prod(poolsize)) 
        # initialize weights with random weights 
        W_bound = numpy.sqrt(6. / (fan_in + fan_out)) 
        self.W = theano.shared( 
            numpy.asarray( 
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), 
                dtype=theano.config.floatX 
            ), 
            borrow=True 
        ) 
~~~

## Theano - Python class for Convolutional Layers with Max Pooling2

~~~python
        # the bias is a 1D tensor -- one bias per output feature map 
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX) 
        self.b = theano.shared(value=b_values, borrow=True) 
 
        # convolve input feature maps with filters 
        conv_out = conv.conv2d( 
            input=input, 
            filters=self.W, 
            filter_shape=filter_shape, 
            image_shape=image_shape 
        ) 
 
        # downsample each feature map individually, using maxpooling 
        pooled_out = downsample.max_pool_2d( 
            input=conv_out, 
            ds=poolsize, 
            ignore_border=True 
        ) 
 
        # add the bias term. Since the bias is a vector (1D array), we first 
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will 
        # thus be broadcasted across mini-batches and feature map 
        # width & height 
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) 
 
        # store parameters of this layer 
        self.params = [self.W, self.b] 
 
        # keep track of model input 
        self.input = input 
~~~

## Theano -- define our CNN training parameters

~~~python
learning_rate=0.1 
dataset='/home/ubuntu/notebook/mnist.pkl.gz' 
nkerns=[100, 50] 
batch_size=128 
~~~

## Theano -- load the MNIST dataset

~~~python
from logistic_sgd import load_data 
 
rng = numpy.random.RandomState(23455) 
 
datasets = load_data(dataset) 
 
train_set_x, train_set_y = datasets[0] 
valid_set_x, valid_set_y = datasets[1] 
test_set_x, test_set_y = datasets[2] 
 
# compute number of minibatches for training, validation and testing 
n_train_batches = train_set_x.get_value(borrow=True).shape[0] 
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] 
n_test_batches = test_set_x.get_value(borrow=True).shape[0] 
n_train_batches /= batch_size 
n_valid_batches /= batch_size 
n_test_batches /= batch_size 
 
n_epochs=10000/n_train_batches 
 
# allocate symbolic variables for the data 
index = T.lscalar()  # index to a [mini]batch 
x = T.matrix('x')   # the data is presented as rasterized images 
y = T.ivector('y')  # the labels are presented as 1D vector of 
                        # [int] labels 
     
print '... data loaded.' 
~~~

## Theano -- define CNN layer-by-layer

~~~python
print '... building the model' 
 
# Reshape matrix of rasterized images of shape (batch_size, 28 * 28) 
# to a 4D tensor, compatible with our LeNetConvPoolLayer 
# (28, 28) is the size of MNIST images. 
layer0_input = x.reshape((batch_size, 1, 28, 28)) 
 
# Construct the first convolutional pooling layer: 
# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24) 
# maxpooling reduces this further to (24/2, 24/2) = (12, 12) 
# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12) 
layer0 = LeNetConvPoolLayer( 
    rng, 
    input=layer0_input, 
    image_shape=(batch_size, 1, 28, 28), 
    filter_shape=(nkerns[0], 1, 25, 25), 
    poolsize=(2, 2) 
) 
 
# Construct the second convolutional pooling layer 
# filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8) 
# maxpooling reduces this further to (8/2, 8/2) = (4, 4) 
# 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4) 
#layer1 = LeNetConvPoolLayer( 
#    rng, 
#    input=layer0.output, 
#    image_shape=(batch_size, nkerns[0], 9, 9), 
#    filter_shape=(nkerns[1], nkerns[0], 4, 4), 
#    poolsize=(2, 2) 
#) 
 
~~~

## Theano -- define CNN layer-by-layer

~~~python
# the HiddenLayer being fully-connected, it operates on 2D matrices of 
# shape (batch_size, num_pixels) (i.e matrix of rasterized images). 
# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4), 
# or (500, 50 * 4 * 4) = (500, 800) with the default values. 
layer2_input = layer0.output.flatten(2) 
 
# construct a fully-connected sigmoidal layer 
layer2 = HiddenLayer( 
    rng, 
    input=layer2_input, 
    n_in=nkerns[0] * 2 * 2, 
    n_out=50, 
    activation=T.tanh 
) 
 
# classify the values of the fully-connected sigmoidal layer 
layer3 = LogisticRegression(input=layer2.output, n_in=50, n_out=10) 
 
# the cost we minimize during training is the NLL of the model 
cost = layer3.negative_log_likelihood(y) 
 
~~~

## Theano -- define CNN layer-by-layer

~~~python
# create a function to compute the mistakes that are made by the model 
test_model = theano.function( 
    [index], 
    layer3.errors(y), 
    givens={ 
        x: test_set_x[index * batch_size: (index + 1) * batch_size], 
        y: test_set_y[index * batch_size: (index + 1) * batch_size] 
    } 
) 
 
validate_model = theano.function( 
    [index], 
    layer3.errors(y), 
    givens={ 
        x: valid_set_x[index * batch_size: (index + 1) * batch_size], 
        y: valid_set_y[index * batch_size: (index + 1) * batch_size] 
    } 
) 
 
~~~

## Theano -- define CNN layer-by-layer

~~~python
# create a list of all model parameters to be fit by gradient descent 
params = layer3.params + layer2.params + layer0.params 
 
# create a list of gradients for all model parameters 
grads = T.grad(cost, params) 
 
# train_model is a function that updates the model parameters by 
# SGD Since this model has many parameters, it would be tedious to 
# manually create an update rule for each model parameter. We thus 
# create the updates list by automatically looping over all 
# (params[i], grads[i]) pairs. 
updates = [ 
    (param_i, param_i - learning_rate * grad_i) 
    for param_i, grad_i in zip(params, grads) 
] 
 
train_model = theano.function( 
    [index], 
    cost, 
    updates=updates, 
    givens={ 
        x: train_set_x[index * batch_size: (index + 1) * batch_size], 
        y: train_set_y[index * batch_size: (index + 1) * batch_size] 
    } 
) 
print '... model built' 

~~~

## Theano -- train the CNN

~~~python
############### 
# TRAIN MODEL # 
############### 
%matplotlib inline 
print '... training' 
# early-stopping parameters 
patience = 10000  # look as this many examples regardless 
patience_increase = 2  # wait this much longer when a new best is 
                           # found 
improvement_threshold = 0.995  # a relative improvement of this much is 
                                   # considered significant 
validation_frequency = min(n_train_batches, patience / 2) 
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 
 
best_validation_loss = numpy.inf 
best_iter = 0 
test_score = 0. 
start_time = timeit.default_timer() 
 
epoch = 0 
done_looping = False 
 
~~~

## Theano -- train the CNN

~~~python
while (epoch < n_epochs) and (not done_looping): 
    epoch = epoch + 1 
    for minibatch_index in xrange(n_train_batches): 
 
        iter = (epoch - 1) * n_train_batches + minibatch_index 
 
        if iter % 100 == 0: 
            print 'training @ iter = ', iter 
        cost_ij = train_model(minibatch_index) 
 
        if (iter + 1) % validation_frequency == 0: 
 
            # compute zero-one loss on validation set 
            validation_losses = [validate_model(i) for i 
                                    in xrange(n_valid_batches)] 
            this_validation_loss = numpy.mean(validation_losses) 
            print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                    (epoch, minibatch_index + 1, n_train_batches, 
                    this_validation_loss * 100.)) 
 
            # if we got the best validation score until now 
            if this_validation_loss < best_validation_loss: 
 
                #improve patience if loss improvement is good enough 
                if this_validation_loss < best_validation_loss *  \ 
                    improvement_threshold: 
                    patience = max(patience, iter * patience_increase) 
 
                # save best validation score and iteration number 
                best_validation_loss = this_validation_loss 
                best_iter = iter 
 
                # test it on the test set 
                test_losses = [ 
                    test_model(i) 
                    for i in xrange(n_test_batches) 
                ] 
                test_score = numpy.mean(test_losses) 
                print(('     epoch %i, minibatch %i/%i, test error of ' 
                        'best model %f %%') % 
                        (epoch, minibatch_index + 1, n_train_batches, 
                        test_score * 100.)) 
                 
~~~

## Theano -- train the CNN

~~~python
    display.clear_output() 
    plt.imshow(tile_raster_images( 
    X = layer0.W.get_value(borrow=True), 
    img_shape=(25,25), 
    tile_shape=(10,10), 
    tile_spacing=(1,1)),  
    cmap= cm.Greys_r, 
    aspect='auto') 
    plt.axis('off') 
    plt.title('Layer 0 convolutional filters, training cost: ' + str(test_score * 100)) 
    plt.show() 
    plt.imshow(layer2.W.get_value(borrow=True)[:,:].T,  
    cmap= cm.Greys_r) 
    plt.axis('off') 
    plt.title('Layer 1 fully connected weights, training cost: ' + str(test_score * 100))    
    plt.show() 
    plt.imshow(layer3.W.get_value(borrow=True)[:,:].T,  
    cmap= cm.Greys_r) 
    plt.axis('off') 
    plt.title('Layer 2 fully connected weights, training cost: ' + str(test_score * 100))    
    plt.show() 
 
    if patience <= iter: 
        done_looping = True 
        break 
 
end_time = timeit.default_timer() 
print('Optimization complete.') 
print('Best validation score of %f %% obtained at iteration %i, ' 
        'with test performance %f %%' % 
        ((1 - best_validation_loss) * 100., best_iter + 1, (1 - test_score) * 100.)) 

~~~

## Torch

* char-rnn 程式與實作
** https://github.com/karpathy/char-rnn
** 資料集
** http://cs.stanford.edu/people/karpathy/char-rnn/
** 研究分析
** http://karpathy.github.io/2015/05/21/rnn-effectiveness/

## Torch 實作

* nvidia-docker
* nvidia-docker run -it kaixhin/cuda-torch:8.0
* cd 
* git clone https://github.com/karpathy/char-rnn.git
* th train.lua
* th sample.lua cv/lm_lstm_epoch9.46_1.4349.t7
* th convert_gpu_cpu_checkpoint.lua cv/lm_lstm_epoch30.00_1.3950.t7
* PyTorch https://github.com/nearai/pytorch-tools
* torch lenet http://blog.csdn.net/u010946556/article/details/51332644
