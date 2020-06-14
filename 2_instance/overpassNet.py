#############################
# Import Required Libraries #
#############################

import sys 
import numpy as np
from make_patches_oneToMany import *
from args import get_parser
from PIL import Image
import os

##########################
# Import Keras Libraries #
##########################
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, ConvLSTM2D, Conv3D, maximum, concatenate
from keras.layers import Input, concatenate, UpSampling2D, Dropout, Reshape, RepeatVector, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD, Adam




def loss1(gt, pred):
  smooth = 1
  # numOfLabelGt = 0
  # numOfLabelPre = 0

  #flatten image patches
  print (K.int_shape(gt))
  print (K.int_shape(pred))
  gt = K.reshape(gt, (-1,6))
  pred = K.reshape(pred, (-1,6))
  print (K.int_shape(gt))
  print (K.int_shape(K.expand_dims(pred, axis=-2))) 
  print (K.int_shape(K.expand_dims(pred, axis=-1))) 
  prod = K.expand_dims(gt, axis=-1) * K.expand_dims(pred, axis=-2)
  print (K.int_shape(prod)) 

  inter = K.sum(prod, axis = 0)
  union = K.expand_dims(K.sum(gt,axis=0),axis=-1) + K.expand_dims(K.sum(pred,axis=0),axis=-2)
  d = -(2. * inter + smooth) / (union + smooth)
  print (K.int_shape(d))
  diag = K.eye(6)
  
  corres_layer = d * diag
  differ_layer = d * (1-diag)

  loss = - (K.sum(differ_layer)/2)
  #  loss = K.sum(corres_layer) - 0.01 * (K.sum(differ_layer)/2)
  #loss = - K.mean(K.sum(differ_layer, axis = -1) / 60)
  #loss = K.sum(corres_layer) - K.sum(differ_layer)


  #print (K.int_shape(d))
  return loss

def loss2(gt, pred):




  gt1 = gt[64,64,1, :]
  gt2 = gt[64,64,2, :]
  gt3 = gt[64,64,3, :]
  gt4 = gt[64,64,4, :]
  gt5 = gt[64,64,5, :]
  gt6 = gt[64,64,6, :]

  pred1 = gt[64,64,1, :]
  pred2 = gt[64,64,2, :]
  pred3 = gt[64,64,3, :]
  pred4 = gt[64,64,4, :]
  pred5 = gt[64,64,5, :]
  pred6 = gt[64,64,6, :]

  #layer1
  prod1_p = K.expand_dims(gt1, axis = -1) * K.expand_dims(pred1, axis = -2)

  l1 = dice_coef(gt2,pred2)
  l2 = dice_coef(gt2,pred1)
  l3 = dice_coef(gt2,pred3)
  l4 = dice_coef(gt2,pred4)
  l5 = dice_coef(gt2,pred5)
  l6 = dice_coef(gt2,pred6)

  emplify = 2

  loss2 = -l1 + emplify * l2 + emplify * l3 + emplify * l4 + emplify * l5
  return loss2



def dice_coef_format2(gt, pred):
  smooth = 1
  prod = K.expand_dims(gt, axis=-1) * K.expand_dims(pred, axis=-2)
  inter = K.sum(prod, axis=1)
  union = K.expand_dims(K.sum(gt,axis=1),axis=-1) + K.expand_dims(K.sum(pred,axis=1),axis=-2)
  d = (2. * inter + smooth) / (union + smooth)
  return d

def dice_coef(y_true, y_pred):
  smooth = 1

  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)

  intersection = K.sum(y_true_f * y_pred_f)

  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)



#Define the neural network
def get_unet(patch_height,patch_width, n_ch):
  inputs = Input((patch_height, patch_width, n_ch))


  # First part
  conv1 = Convolution2D(16, (3, 3), activation='relu', padding='same')(inputs)

  conv1 = Dropout(0.2)(conv1)
  conv1 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Convolution2D(32, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Dropout(0.2)(conv2)
  conv2 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Dropout(0.2)(conv3)
  conv3 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv3)

  up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2],  axis=-1)
  conv4 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up1)
  conv4 = Dropout(0.2)(conv4)
  conv4 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv4)

  up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1],  axis=-1)
  conv5 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up2)
  conv5 = Dropout(0.2)(conv5)
  conv5 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv5)

  #Seconde part
  conv12 = Convolution2D(16, (3, 3), activation='relu', padding='same')(inputs)

  conv12 = Dropout(0.2)(conv12)
  conv12 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv12)
  pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)

  conv22 = Convolution2D(32, (3, 3), activation='relu', padding='same')(pool12)
  conv22 = Dropout(0.2)(conv22)
  conv22 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv22)
  pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)

  conv32 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool22)
  conv32 = Dropout(0.2)(conv32)
  conv32 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv32)

  up12 = concatenate([UpSampling2D(size=(2, 2))(conv32), conv22],  axis=-1)
  conv42 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up12)
  conv42 = Dropout(0.2)(conv42)
  conv42 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv42)

  up22 = concatenate([UpSampling2D(size=(2, 2))(conv42), conv12],  axis=-1)
  conv52 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up22)
  conv52 = Dropout(0.2)(conv52)
  conv52 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv52)


    #Third part

  conv13 = Convolution2D(16, (3, 3), activation='relu', padding='same')(inputs)

  conv13 = Dropout(0.2)(conv13)
  conv13 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv13)
  pool13 = MaxPooling2D(pool_size=(2, 2))(conv13)

  conv23 = Convolution2D(32, (3, 3), activation='relu', padding='same')(pool13)
  conv23 = Dropout(0.2)(conv23)
  conv23 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv23)
  pool23 = MaxPooling2D(pool_size=(2, 2))(conv23)

  conv33 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool23)
  conv33 = Dropout(0.2)(conv33)
  conv33 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv33)

  up13 = concatenate([UpSampling2D(size=(2, 2))(conv33), conv23],  axis=-1)
  conv43 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up13)
  conv43 = Dropout(0.2)(conv43)
  conv43 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv43)

  up23 = concatenate([UpSampling2D(size=(2, 2))(conv43), conv13],  axis=-1)
  conv53 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up23)
  conv53 = Dropout(0.2)(conv53)
  conv53 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv53)

      #Fourth part

  conv14 = Convolution2D(16, (3, 3), activation='relu', padding='same')(inputs)

  conv14 = Dropout(0.2)(conv14)
  conv14 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv14)
  pool14 = MaxPooling2D(pool_size=(2, 2))(conv14)

  conv24 = Convolution2D(32, (3, 3), activation='relu', padding='same')(pool14)
  conv24 = Dropout(0.2)(conv24)
  conv24 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv24)
  pool24 = MaxPooling2D(pool_size=(2, 2))(conv24)

  conv34 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool24)
  conv34 = Dropout(0.2)(conv34)
  conv34 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv34)

  up14 = concatenate([UpSampling2D(size=(2, 2))(conv34), conv24],  axis=-1)
  conv44 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up14)
  conv44 = Dropout(0.2)(conv44)
  conv44 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv44)

  up24 = concatenate([UpSampling2D(size=(2, 2))(conv44), conv14],  axis=-1)
  conv54 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up24)
  conv54 = Dropout(0.2)(conv54)
  conv54 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv54)


    #Fifth part

  conv15 = Convolution2D(16, (3, 3), activation='relu', padding='same')(inputs)

  conv15 = Dropout(0.2)(conv15)
  conv15 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv15)
  pool15 = MaxPooling2D(pool_size=(2, 2))(conv15)

  conv25 = Convolution2D(32, (3, 3), activation='relu', padding='same')(pool15)
  conv25 = Dropout(0.2)(conv25)
  conv25 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv25)
  pool25 = MaxPooling2D(pool_size=(2, 2))(conv25)

  conv35 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool25)
  conv35 = Dropout(0.2)(conv35)
  conv35 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv35)

  up15 = concatenate([UpSampling2D(size=(2, 2))(conv35), conv25],  axis=-1)
  conv45 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up15)
  conv45 = Dropout(0.2)(conv45)
  conv45 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv45)

  up25 = concatenate([UpSampling2D(size=(2, 2))(conv45), conv15],  axis=-1)
  conv55 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up25)
  conv55 = Dropout(0.2)(conv55)
  conv55 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv55)

  #Six part

  conv16 = Convolution2D(16, (3, 3), activation='relu', padding='same')(inputs)

  conv16 = Dropout(0.2)(conv16)
  conv16 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv16)
  pool16 = MaxPooling2D(pool_size=(2, 2))(conv16)

  conv26 = Convolution2D(32, (3, 3), activation='relu', padding='same')(pool16)
  conv26 = Dropout(0.2)(conv26)
  conv26 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv26)
  pool26 = MaxPooling2D(pool_size=(2, 2))(conv26)

  conv36 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool26)
  conv36 = Dropout(0.2)(conv34)
  conv36 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv36)

  up16 = concatenate([UpSampling2D(size=(2, 2))(conv36), conv26],  axis=-1)
  conv46 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up16)
  conv46 = Dropout(0.2)(conv46)
  conv46 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv46)

  up26 = concatenate([UpSampling2D(size=(2, 2))(conv46), conv16],  axis=-1)
  conv56 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up26)
  conv56 = Dropout(0.2)(conv56)
  conv56 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv56)


  #concatenate

  squeeze_conv = Convolution2D(1,(1,1), activation='sigmoid')
  conv5 = squeeze_conv(conv5)
  conv52 = squeeze_conv(conv52)
  conv53 = squeeze_conv(conv53)
  conv54 = squeeze_conv(conv54)
  conv55 = squeeze_conv(conv55)
  conv56 = squeeze_conv(conv56)

  concatenate7 = concatenate([conv5,conv52],  axis = -1)
  concatenate7 = concatenate([concatenate7,conv53],  axis = -1)
  concatenate7 = concatenate([concatenate7,conv54],  axis = -1)
  concatenate7 = concatenate([concatenate7,conv55],  axis = -1)
  concatenate7 = concatenate([concatenate7,conv56],  axis = -1)

  concatenate8 = maximum([conv5, conv52, conv53, conv54, conv55, conv56])

  model = Model(input=inputs, output=[conv5,conv52,conv53,conv54,conv55,conv56,concatenate8])
 # model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=['accuracy'])


  model.compile(optimizer=Adam(lr=1e-4), \
                loss = [dice_coef_loss,dice_coef_loss,dice_coef_loss,dice_coef_loss,dice_coef_loss,dice_coef_loss,dice_coef_loss], \

                loss_weights = [1, 1, 1, 1, 1, 1, 1], \

                metrics=['accuracy'])
  return model




def main(args):
  mode = args.mode


  
  input_path = args.input_path
  output_path = args.output_path
  weights_path = args.weights_path

  trainImgPath = input_path + args.trainImgPath
  trainGrPath = input_path + args.trainGrPath
  trainImgDt = args.trainImgDt
  trainGrDt = args.trainGrDt

  data_type = args.data_type
  patch_size = (args.patch_size, args.patch_size)    

  if mode == 'train':

    imgs_train, masks_train = get_my_patches(trainImgPath,trainGrPath, trainImgDt, trainGrDt, patch_size, recursive=True)

    imgs_train = imgs_train.astype('float32')

    masks_train = masks_train.astype('float32')
    
  
    m1, m2, m3, m4, m5, m6, m7 = np.split(masks_train, 7, axis = 3 )

    first_six = np.concatenate((m1,m2,m3,m4,m5,m6), axis = -1)
  
    #print first_six.shape
    #print m1.shape
    
    model = get_unet(patch_size[0], patch_size[1], 1)

    ckpt = ModelCheckpoint(filepath=weights_path, verbose=1, monitor='val_loss', mode='auto', save_best_only=True)
    
    model.fit(imgs_train, [m1, m2, m3, m4, m5, m6, m7], batch_size=64, epochs=5, verbose=1, shuffle=True, validation_split=0.1, callbacks=[ckpt])

    
  
  
  elif mode == 'predict':
    print ('predicting')
    
    img_paths = get_images_pre(input_path, data_type, recursive=True)
    #img_paths = ['./imcurve.tif','./imgauss.tif']
    for i, img_path in enumerate(img_paths):

      path_list = img_path.split(os.sep)
      name_the_file = (path_list[-1].split('.'))[0]

      print(name_the_file)      
      img = Image.open(img_path).convert('L')
      img.save(output_path + name_the_file + '.png','PNG')

      img = np.array(img).astype('float32')

  
      img = img.reshape(1, img.shape[0], img.shape[1], 1)

      
      model = get_unet(*img.shape[1:])
      model.load_weights(weights_path)
      
      o1,o2,o3,o4,o5,o6,o7= model.predict(img)

      o1 = o1[0,:,:,0] * 255.0
      o1 = Image.fromarray(o1).convert('L')

      o2 = o2[0,:,:,0] * 255.0
      o2 = Image.fromarray(o2).convert('L')    

      o3= o3[0,:,:,0] * 255.0
      o3 = Image.fromarray(o3).convert('L')          

      o4 = o4[0,:,:,0] * 255.0
      o4 = Image.fromarray(o4).convert('L')  

      o5 = o5[0,:,:,0] * 255.0
      o5 = Image.fromarray(o5).convert('L')  

      o6 = o6[0,:,:,0] * 255.0
      o6 = Image.fromarray(o6).convert('L')  

      o7 = o7[0,:,:,0] * 255.0
      o7 = Image.fromarray(o7).convert('L')  


      

      o1.save(output_path + name_the_file + '_branch_' + str(1) + '.png','PNG')
      o2.save(output_path + name_the_file + '_branch_' + str(2) + '.png', 'PNG')    
      o3.save(output_path + name_the_file + '_branch_' + str(3) + '.png', 'PNG')    
      o4.save(output_path + name_the_file + '_branch_' + str(4) + '.png', 'PNG')    
      o5.save(output_path + name_the_file + '_branch_' + str(5) + '.png','PNG')
      o6.save(output_path + name_the_file + '_branch_' + str(6) + '.png', 'PNG')
      o7.save(output_path + name_the_file + '_branches_merge' + '.png', 'PNG')

if __name__ == "__main__":
    print ("running")
    args = get_parser()
    main(args)