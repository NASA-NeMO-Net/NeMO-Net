"""
@author: vijay
implementation of superresolution model from http://vllab.ucmerced.edu/wlai24/LapSRN/.
Currently this code only works for a one-level model which means the variables hr_fac and upsamp_fac should be the same.
It can be extended easily to multiple levels by tweaking a bit.
"""

# import libraries
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np, time, glob, scipy.io as scio
from random import shuffle
from scipy.misc import imresize

from keras.layers import Input, concatenate, Add, LeakyReLU, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Nadam
import keras.backend as K

# specify root directory. All other folders (trainA, trainB etc) are addressed with respect to this.
rootDir = '/home/vijay/Desktop/coralWork/data/sentinel_wv/'

# unique string added to saved files for identification
fadd = "_".join( [ str(k) for k in list(time.localtime())[:5] ])
fadd = fadd + '_lapSRN_sentinel_'

# some default values
scaling_fac = 255.0
channel_axis = -1
channel_first = False

# if pretrained model needs to be loaded, use load_models = 1 and specify a path for the model
load_models = 0
load_fname = rootDir + 'save_models/' + '2018_5_18_7_30_lapSRN_sentinel_3_4_2_64_32_4_4_1_100_4_epoch_1638_valacc_0.03059_.h5'

# model parameters
mo = 0.9 # momentum for batchnorm layer
ker = 3 # conv kernel size
num_conv_inconvset = 4 # num conv layers within each convset
num_convset_inlevel = 2 # num repeated convsets within each level
nfilters = 64 # number of filters in conv layer
nch = 4 # num channels to use (3 if only RGB, 4 to include NIR)
batchsize = 32 
epochs = 50000 # total num of graident updates (not epochs as conventionally defined in neural nets)

# specify validation parameters
# After every "val_interval" number of gradient updates,
# validate on n_val_iter*batchsize_val examples and calculate mean accuracy to check if model's accuracy improved
val_interval = 1
batchsize_val = batchsize 
n_val_iter = 10

# a model is saved after every "save_interval" gradient updates
save_interval = 100

# super resolution scaling factors
hr_fac = 4 # total upsampling factor (low-res to hi-res)
upsamp_fac = 4 # factor by SRN model upsamples at each level
if hr_fac != upsamp_fac:
    print("hr_fac and upsamp_fac not the same. This version of the code is not supported to handle this.")
    sdfgdfg
    
n_levels = int( np.log2(hr_fac) / np.log2(upsamp_fac) ) # number of levels needed to upsamplehr_fac
source_img_shape = (60, 60, nch) # source image shape
tar_img_shape = (60*hr_fac, 60*hr_fac, nch) # target image shape

# derived filename identifier to add to saved models
pars = [ker, 
        num_conv_inconvset, 
        num_convset_inlevel, 
        nfilters, 
        batchsize, 
        upsamp_fac, 
        hr_fac, 
        val_interval, 
        save_interval] 

fadd = fadd + "_".join([str(k) for k in pars])+'_'

# turn off interactive plotting
plt.ioff()

#%% model functions

def inception_layer(input_img, nfilters):
    '''
    This function creates a modified inception layer. Refer to structure of inception layer from the literature.
    A minor difference is that here I have a conv2D after the concatenate step to make sure
    the output tensor is same sized as the input.
    '''
    branch_1 = Conv2D(nfilters, kernel_size = 1, padding = 'same')(input_img)
    branch_1 = BatchNormalization(momentum = mo)(branch_1)
    branch_1 = LeakyReLU(alpha = 0.2)(branch_1)
    branch_1 = Conv2D(nfilters, kernel_size = 3, padding = 'same')(branch_1)
    branch_1 = BatchNormalization(momentum = mo)(branch_1)
    branch_1 = LeakyReLU(alpha = 0.2)(branch_1)
    
    branch_2 = Conv2D(nfilters, kernel_size = 1, padding = 'same')(input_img)
    branch_2 = BatchNormalization(momentum = mo)(branch_2)
    branch_2 = LeakyReLU(alpha = 0.2)(branch_2)
    branch_2 = Conv2D(nfilters, kernel_size = 6, padding = 'same')(branch_2)
    branch_2 = BatchNormalization(momentum = mo)(branch_2)
    branch_2 = LeakyReLU(alpha = 0.2)(branch_2)
    
    branch_3 = Conv2D(nfilters,kernel_size = 3, padding = 'same')(input_img)
    branch_3 = BatchNormalization(momentum = mo)(branch_3)
    branch_3 = LeakyReLU(alpha = 0.2)(branch_3)
    branch_3 = Conv2D(nfilters, kernel_size = 1, padding = 'same')(branch_3)
    branch_3 = BatchNormalization(momentum = mo)(branch_3)
    branch_3 = LeakyReLU(alpha = 0.2)(branch_3)
    
    out = concatenate([branch_1, branch_2, branch_3], axis = -1)
    out = Conv2D(nfilters,kernel_size = ker,strides = 1,padding = 'same')(out)
    
    return out

def convset(num_conv_inconvset, inp = Input(shape = (None, None, nfilters))):
    '''
    This function creates a set of "num_conv_inconvset" conv2D layers.
    '''
    H  =  inp
    for i in np.arange(num_conv_inconvset):
        H = Conv2D(nfilters, kernel_size = ker, strides = 1, padding = 'same')(H)
        H = BatchNormalization(momentum = mo)(H)
        H = LeakyReLU(alpha = 0.2)(H)
    return Model(inp, H)

def feat_extract_singlelevel(inp = Input(shape = (None, None, nfilters))):
    '''
    Creates one level of model (each level upsamples by a factor of upsamp_fac)
    '''
    H = inp
    convset_model = convset(num_conv_inconvset)
    skip = inp # skip connection
    for i in np.arange(num_convset_inlevel):
        H = convset_model(H)
        H = Add()([H, skip])
    H = Conv2DTranspose(nfilters, kernel_size = ker, strides = upsamp_fac, padding = 'same')(H)
    H = BatchNormalization(momentum = mo)(H)
    out = LeakyReLU(alpha = 0.2)(H)
    
    # calc residual (see paper)
    H = Conv2D(nch, kernel_size = ker, strides = 1, padding = 'same')(out)
    res = LeakyReLU(alpha = 0.2)(H)
    
    return Model(inp, [res, out])

def build_lap_model(n_levels):

    inp = Input(shape = source_img_shape, name = 'LR_input')
    
    feat_extract_model = feat_extract_singlelevel()

    H = inp
    H = inception_layer(H, nfilters)
    output = []
    inp1 = H
    inp2 = inp
    for level in np.arange(n_levels):
        res, out = feat_extract_model(inp1)
        
        # reconstruction branch
        H = Conv2DTranspose(nch, 
                            kernel_size = ker, 
                            strides = upsamp_fac, 
                            padding = 'same', 
                            kernel_initializer = init_kernel)(inp2)
        H = Add()([H, res])
        rec = H
        output.append(rec)
        
        inp1 = out
        inp2 = rec
    
    model = Model(inputs = inp, outputs = output)
    model.summary()
    return model

#%% functions for custom initializer
class CustomInitializer:
    '''This function along with init_kernel(), upsample_filt() and bilinear_upsample_weights() defined below
    are used to initialize the conv2DTranspose parmaeters to match bilinear upsampling. The idea is to 
    start with bilinear and learn the parameters to fit to the data.
    '''
    def __call__(self, shape, dtype = None):
        return init_kernel(shape, dtype = dtype)

def init_kernel(shape, dtype = None):
    ker = bilinear_upsample_weights(upsamp_fac, shape[-1])
    return ker

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = 2 * factor - factor % 2
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype = np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in np.arange(number_of_classes):
        
        weights[:, :, i, i] = upsample_kernel
    
    return weights

#%% data processing functions

def rescale(dataset):
    depth = 255.
    dataset_norm = (dataset.astype(np.float32) - depth/2) / (depth/2)
    return dataset_norm

def re_rescale(dataset):
    '''
    This function reverses rescale()
    '''
    depth = 255.
    dataset_norm = dataset.astype(np.float32) * (depth/2) +  depth/2
    return dataset_norm

def load_files(file_pattern):
    return glob.glob(file_pattern) 

def bicubic_upsampling_channelwise(img, size):
    '''
    This function takes a low-res img and upsamples (independently done for each channel) using bicubic.
    The upsampling factor is specified as "size". Refer to scipy's imresize function.
    '''
    nch = img.shape[-1]
    temp = np.zeros(tuple([int(k*size) for k in img.shape[:-1]]) + (nch,))
    for ch in range(nch):
        temp[:,:,ch] = imresize(img[:,:,ch], interp = 'bicubic', size = size)
    return temp

#%% loss and metric functions
    
def PSNRLoss(y_true, y_pred):
    '''
    calculates PSNR (peak signal to noise ratio) useful for semantic segmentation.
    '''
    return -1*(-10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.))

def psnr(y_true, y_pred):
    '''
    Numpy version of PSNRLoss()
    '''
    return -1*(-10. * np.log10(np.mean(np.square(y_pred - y_true))) / np.log(10.))

def charbonnierLoss(y_true, y_pred):
    '''
    calculates charbonnier loss useful for semantic segmentation.
    '''
    return K.mean(K.sqrt(K.square(y_pred - y_true) + K.constant(1e-6)), axis = -1)

def mse(ytrue, ypred):
    '''
    calculates mean squared error metric.
    '''
    ytrue = np.array(ytrue,dtype = float)
    ypred = np.array(ypred,dtype = float)
    return np.sqrt(np.mean((ytrue - ypred) ** 2))

#%% load and save functions
    
def gen_minibatch(dataA, dataB, batchsize, source_img_shape, tar_img_shape):
    '''
    Generator function that yields matching high-res and lo-res images.
    dataA and dataB are lists of filenames from high-res and lo-res image directories.
    For every call, a "batchsize" number of images are returned randomly from the corresponding directories.
    The images are also normalized using the rescale() function.
    '''
    length = len(dataA)
    epoch = i = 0
    ndata = None
    
    while True:
        size = ndata if ndata else batchsize
       
        if i+size > length:
            c = list( zip(dataA, dataB) )
            shuffle(c)
            dataA, dataB = zip(*c)
            i = 0
            epoch += 1
            
        hr_images = np.empty((0,) + tar_img_shape)
        lr_images = np.empty((0,) + source_img_shape)
        
        for j in range(i, i + size):
            
            # load data
            data_hr = scio.loadmat(dataA[j])['data'][:, :, :nch]
            data_lr = scio.loadmat(dataB[j])['data'][:, :, :nch]
            
            # append loaded data to array
            hr_images = np.concatenate((hr_images, np.reshape(data_hr, (1,) + tar_img_shape)), axis = 0)
            lr_images = np.concatenate((lr_images, np.reshape(data_lr, (1,) + source_img_shape)), axis = 0)
        
        # normalize
        hr_images = rescale(hr_images)
        lr_images = rescale(lr_images)
        
        i += size
        ndata = yield epoch, hr_images, lr_images

def save_imgs(epoch):
    '''
    Call this function to compare images generated by the model and baseline (the ones I use for reporting).
    It generates and saves 5 image samples for ground truth, model, baseline, low-res. 
    
    '''
    r, c = 4, min(5, batchsize) # 4 rows (ground truth, model, baseline, low-res) and 5 columns (different image samples)

    # generate validation images
    e_, hr_imgs_valid, lr_imgs_valid = next(batch_valid_combined)
    
    # super-resolution prediction
    val_pred = model.predict(lr_imgs_valid, batch_size = batchsize_valA)
    
    # although n_levels > 1 is not supported, I wanted to leave it here if you wanted to extend to multiple levels.
    imsize = hr_imgs_valid.shape[1]
    for level in np.arange(n_levels):
        
        if n_levels == 1:
            gen_imgs = val_pred
        else:
            gen_imgs = val_pred[n_levels - level - 1]
        
        gen_imgs = np.array(re_rescale(gen_imgs), dtype = np.uint8) # normalized predicted images
        
        # baseline prediction
        val_pred_bicubic = np.zeros((batchsize_valA, imsize, imsize, nch))
        for i in np.arange(batchsize_valA):
            val_pred_bicubic[i, :, :, :] = bicubic_upsampling_channelwise( img = lr_imgs_valid[i], size = 1.*( upsamp_fac**(n_levels-level) ) )
        
        imsize = int(imsize / upsamp_fac)
        
        # print accuracies of the model and baseline
        print("Level %d mse for model is %f\n" %(n_levels - level, mse(hr_imgs_valid, gen_imgs)))
        print("Level %d mse for baseline is %f\n" %(n_levels - level, mse(hr_imgs_valid, val_pred_bicubic)))
        
        # plotting
        fig, axs = plt.subplots(r, c)
        cnt = 0
        
        for j in range(c):
            
            axs[0,j].imshow(np.array(re_rescale(hr_imgs_valid[cnt, :,:,:nch-1]), dtype = np.uint8), plt.get_cmap('jet'))
            axs[0,j].axis('off')
            axs[0,j].set_title('ground \ntruth', fontsize = 6)
            
            axs[1,j].imshow(gen_imgs[cnt, :,:,:nch-1], plt.get_cmap('jet'))
            axs[1,j].axis('off')
            axs[1,j].set_title('SR model', fontsize = 6)
            
            axs[2,j].imshow(np.array(val_pred_bicubic[cnt, :, :, :nch-1], dtype = np.uint8), plt.get_cmap('jet'))
            axs[2,j].axis('off')
            axs[2,j].set_title('baseline', fontsize = 6)
            
            axs[3,j].imshow(np.array(re_rescale(lr_imgs_valid[cnt, :, :, :nch-1]), dtype = np.uint8), plt.get_cmap('jet'))
            axs[3,j].axis('off')
            axs[3,j].set_title('LR input', fontsize = 6)
            
            cnt += 1
        
        # save figure
        fig.savefig(rootDir + "vis_images/" + fadd + "_"+str(n_levels - level) + "_%d_epoch_%d.png" % (hr_fac, epoch), dpi = 500)
        plt.close(fig)
        
#%% specify HR and LR data
# A is HR and B is LR

train_A = load_files(rootDir + 'trainA/' + '*.mat') # has filenames
train_B = load_files(rootDir + 'trainB/' + '*.mat') # has filenames
batch_train_combined = gen_minibatch(train_A, train_B, batchsize, source_img_shape, tar_img_shape)

valid_A = load_files(rootDir + 'testA/' + '*.mat') # has filenames
valid_B = load_files(rootDir + 'testB/' + '*.mat') # has filenames
batchsize_valA = min(len(valid_A), batchsize_val)
batchsize_valB = min(len(valid_B), batchsize_val)
batch_valid_combined = gen_minibatch(valid_A, valid_B, batchsize_valA, source_img_shape, tar_img_shape)

#%% model train or load

if load_models == 1:

    # load a saved model
    from keras.models import load_model
    fname = load_fname
    model = load_model(fname, custom_objects = {'charbonnierLoss':charbonnierLoss, 'PSNRLoss':PSNRLoss, 'init_kernel':CustomInitializer})

else:
    
    # build LAPSRN model
    model = build_lap_model(n_levels)
    
    # specify losses for each level
    losses = [charbonnierLoss] * n_levels
    
    # specify loss weights for each level
    loss_wt = [1] * n_levels
    
    # compile model
    model.compile(loss = losses, loss_weights = loss_wt, optimizer = Nadam(lr = 1e-4))
    
    #%% train
    
    loss = [] # container to store loss after every gradient update
    
    val_acc = 1e10 # set some high value
    
    for epoch in range(epochs):
        
        # Generate training images
        e_, hr_imgs, lr_imgs = next(batch_train_combined)
        
        # do one update
        model_loss = model.train_on_batch(lr_imgs, hr_imgs)
        
        # save loss for plotting
        print(epoch, model_loss)
        loss.append([model_loss])
        
        # test disc on validation set 
        if epoch % val_interval == 0:
            
            acc_val = []
            
            for validx in np.arange(n_val_iter):
                
                # generate validation images
                e_, hr_imgs_valid, lr_imgs_valid = next(batch_valid_combined)
                
                # calculate MSE 
                acc_val.append( float("{0:.4f}".format( mse(hr_imgs_valid, model.predict(lr_imgs_valid)) )) )
            
            # mean MSE for validation set
            model_acc_valid = np.mean(acc_val)
            
            # save model if current model is better than previously saved one
            if model_acc_valid < val_acc:
                save_fname = rootDir + 'save_models/' + fadd + str(hr_fac) + '_epoch_' + str(epoch) + '_valacc_' + str(model_acc_valid) + '_'
                model.save(save_fname + '.h5')
                save_imgs(epoch)
                val_acc = model_acc_valid
        
        # If at save interval => save generated image samples and the model (irrespective of its performance getting better)
        if epoch % save_interval == 0:
            save_fname = rootDir + 'save_models/' + fadd + str(hr_fac) + '_epoch_' + str(epoch) + '_valacc_' + str(model_acc_valid) + '_'
            model.save(save_fname + '.h5')
            save_imgs(epoch)
        
        plt.close("all")
    
    # plot losses
    plt.figure()
    plt.plot([k[0][0] for k in loss])
    plt.hold
    plt.plot([k[1][0] for k in loss],'g')
    plt.show()