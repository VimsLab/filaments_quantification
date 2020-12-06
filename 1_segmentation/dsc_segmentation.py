#############################
# Import Required Libraries #
#############################
from __future__ import print_function
import os, sys, fnmatch, argparse
import numpy as np
import io
from make_patches_MT import *
from args import get_parser
from PIL import Image
import cv2
import imageio
from skimage import io
from tqdm import tqdm
from tifffile import TiffFile
##########################
# Import Keras Libraries #
##########################
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Input, UpSampling2D, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.optimizers import SGD, Adam



import tensorflow as tf
print(tf.config.experimental.list_physical_devices('GPU'))
# Specify which GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

def dice_coef(y_true, y_pred):
  smooth = 1.
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)

  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)

def multi_dice_coef_loss(y_true, y1_pred, y2_pred, y3_pred):
  return -0.5 * dice_coef(y_true, y1_pred) - 0.7 * dice_coef(y_true, y2_pred) - dice_coef(y_true, y3_pred)

#########################
# DSC-Unet Architecture #
#########################

#  cross
def get_unet(patch_height,patch_width, n_ch):
  inputs = Input((patch_height, patch_width, n_ch))
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  #conv1 = BatchNormalization()(conv1)
  conv1 = Dropout(0.2)(conv1)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Dropout(0.2)(conv2)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
  #conv2 = BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  #conv3 = BatchNormalization()(conv3)
  conv3 = Dropout(0.2)(conv3)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

  up1 = Concatenate(axis = -1)([UpSampling2D(size=(2, 2))(conv3), conv2])
  conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
  conv4 = Dropout(0.2)(conv4)
  conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

  up2 = Concatenate(axis = -1)([UpSampling2D(size=(2, 2))(conv4), conv1])
  conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
  #conv5 = BatchNormalization()(conv5)
  conv5 = Dropout(0.2)(conv5)
  conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)




  conv21 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
  #conv21 = BatchNormalization()(conv21)
  conv21 = Dropout(0.2)(conv21)
  conv21 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv21)
  pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)

  conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool21)
  #conv22 = BatchNormalization()(conv22)
  conv22 = Dropout(0.2)(conv22)
  conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv22)
  pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)

  conv23 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool22)
  #conv23 = BatchNormalization()(conv23)
  conv23 = Dropout(0.2)(conv23)
  conv23 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv23)

  up21 = Concatenate(axis = -1)([UpSampling2D(size=(2, 2))(conv23), conv22, conv2])
  conv24 = Conv2D(64, (3, 3), activation='relu', padding='same')(up21)
  #conv24 = BatchNormalization()(conv24)
  conv24 = Dropout(0.2)(conv24)
  conv24 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv24)

  up22 = Concatenate(axis = -1)([UpSampling2D(size=(2, 2))(conv24), conv21, conv1])
  conv25 = Conv2D(32, (3, 3), activation='relu', padding='same')(up22)
  conv25 = Dropout(0.2)(conv25)
  conv25 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv25)



  conv31 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv25)
  conv31 = Dropout(0.2)(conv31)
  conv31 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv31)
  pool31 = MaxPooling2D(pool_size=(2, 2))(conv31)

  conv32 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool31)
  conv32 = Dropout(0.2)(conv32)
  conv32 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv32)
  pool32 = MaxPooling2D(pool_size=(2, 2))(conv32)

  conv33 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool32)
  conv33 = Dropout(0.2)(conv33)
  conv33 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv33)

  up31 = Concatenate(axis = -1)([UpSampling2D(size=(2, 2))(conv33), conv32, conv22, conv2])
  conv34 = Conv2D(64, (3, 3), activation='relu', padding='same')(up31)
  conv34 = Dropout(0.2)(conv34)
  conv34 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv34)

  up32 = Concatenate(axis = -1)([UpSampling2D(size=(2, 2))(conv34), conv31, conv21, conv1])
  conv35 = Conv2D(32, (3, 3), activation='relu', padding='same')(up32)
  conv35 = Dropout(0.2)(conv35)
  conv35 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv35)

  squeeze_conv = Conv2D(1,(1,1), activation='sigmoid')
  conv7 = squeeze_conv(conv5)
  conv27 = squeeze_conv(conv25)
  conv37 = squeeze_conv(conv35)

  model = Model(inputs, [conv7, conv27, conv37])
  model.compile(optimizer=Adam(lr=1e-4), loss = dice_coef_loss, loss_weights = [0.2, 0.3, 0.5], metrics=['accuracy'])
  print ('DSC-Unet Loaded')
  return model


def training_set_generator(training_list,batch_size,patch_dims):

    while True:
        np.random.shuffle(training_list)
        num_slices = len(training_list) // batch_size
        for slice in range(num_slices):
            imgs_train = np.zeros((batch_size, patch_dims[0], patch_dims[1], 1),dtype='float32')
            mask_train = np.zeros((batch_size, patch_dims[0], patch_dims[1], 1),dtype='float32')
            for element in range(batch_size):
                training_list_id = slice * batch_size + element
                img, mask = read_images(training_list[training_list_id][0], training_list[training_list_id][1])
                ############
                meanVal = np.mean(img)
                stdVal = np.std(img)
                img -= meanVal
                img /= stdVal
                ############

                imgs_train[element, :, :, 0] = img
                mask_train[element, :, :, 0] = mask

            yield (imgs_train, [mask_train, mask_train, mask_train])

def prediction_set_generator(tiffInput,batch_size):

    while True:
        constant_batch_size = batch_size
        num_slices = (len(tiffInput) // batch_size) + 1
        left_images = len(tiffInput) % batch_size

        h, w = tiffInput.shape[1:] # tiffshape(timestep, h, w)
        for slice in range(num_slices):
            if slice == num_slices - 1:
                if left_images != 0:
                    batch_size = left_images
                else:
                    batch_size = 0

            imgs_test = np.zeros((batch_size, h, w, 1),dtype='float32')

            for element in range(batch_size):
                timestepId = slice * constant_batch_size + element
                mt_mip = tiffInput[timestepId, :, :].astype('float32')
                mt_mip /= 255
                # mt_mip = cv2.GaussianBlur(mt_mip, (5, 5), 0)

                 # percent of original size

                # dim = (width, height)
                # img = mt_mip
                img = mt_mip
                og_img = np.array(img)
                img /= np.max(img)

#meanVal = np.mean(img)
#stdVal = np.std(img)

                meanVal =  0.10874228924512863159
                stdVal =  0.13627362251281738281
                img -= meanVal
                img /= stdVal

                original_img = np.array(mt_mip)

                try:
                    mt_mip -= np.mean(original_img)
                    mt_mip /= np.std(original_img)
                except:
                    import pdb; pdb.set_trace()

                imgs_test[element, :, :, 0] = img

            yield (imgs_test)


def get_training_list(path, extension):
    img_paths = []
    mask_paths = []
    print (path)
    for root, directories, filenames in os.walk(path + '/img_list/'):
        for filename in fnmatch.filter(filenames, extension):
            img_paths.append(os.path.join(root,filename))

    for root, directories, filenames in os.walk(path + '/mask_list/'):
        for filename in fnmatch.filter(filenames, extension):
            mask_paths.append(os.path.join(root,filename))
    img_paths.sort()
    mask_paths.sort()
    img_paths = np.array(img_paths)
    mask_paths = np.array (mask_paths)

    training_list = np.array([img_paths, mask_paths])
    training_list = training_list.transpose()

    return training_list

def read_images (img_path_1, img_path_2):
    img1 = Image.open(img_path_1).convert('L') #grayscale
    img2 = Image.open(img_path_2).convert('L')
    img1 = np.array(img1).astype('float32')
    img2 = np.array(img2).astype('float32')

    img2 /= 255.
    img1 /= 255.
    return np.array(img1), np.array(img2)

def main(args):

  mode = args.mode
  input_path = args.input_path
  output_path = args.output_path
  weights_path = args.weights_path
  data_type = args.data_type
  patch_size = (args.patch_size, args.patch_size)
  batch_size = args.batch_size
  nepochs = args.nepochs

  if mode == 'train':

    training_list = get_training_list(input_path, data_type)

    total_length = len(training_list)

    validation_split = 0.1
    train_data_length = int (total_length * (1 - validation_split))
    validation_data_length = int (total_length * validation_split)
    total_length = int (total_length)
    print("Creating Generators...")

    train_data_gen = training_set_generator(training_list[:train_data_length],batch_size, patch_size)
    val_data_gen = training_set_generator(training_list[train_data_length:],batch_size, patch_size)
    val_steps = (total_length - train_data_length)//batch_size

    print("Beginning Training...")

    model = get_unet(patch_size[0], patch_size[1], 1)
    # from tensorflow.compat.v2.keras.utils import multi_gpu_model
    # model = multi_gpu_model(model, gpus=2)
    ckpt = ModelCheckpoint(filepath=weights_path, verbose=1, monitor='val_loss', mode='auto', save_best_only=True)
    model.fit_generator(train_data_gen, train_data_length//batch_size, epochs= nepochs, verbose=1, callbacks=[ckpt], validation_data=val_data_gen, validation_steps=val_steps)


  elif mode == 'predict':

    img_paths = get_images_pre(input_path, extension=data_type, recursive=True)


    if not os.path.exists(output_path + 'predict'):
      os.makedirs(output_path + 'predict')
    if not os.path.exists(output_path + 'original'):
      os.makedirs(output_path + 'original')
    if not os.path.exists(output_path + 'merged'):
      os.makedirs(output_path + 'merged')

    for i, img_path in enumerate(img_paths):

      img = io.imread(img_path).astype('float32')

      img = np.array(img).astype('float32')
      img /= 255.

      # import pdb; pdb.set_trace()
      # img = np.array(img).astype('float32')
      # img /= np.max(img)
      # img = img * 255.
      # img = Image.fromarray(img).convert('L')
      # img.save(input_path+ '/../' + '_original.png', 'PNG')
      # img = cv2.GaussianBlur(img, (5, 5), 0)
      # img = cv2.medianBlur(img, 5)
      scale_percent = 100 # percent of original size
      width = int(img.shape[1] * scale_percent / 100)
      height = int(img.shape[0] * scale_percent / 100)
      dim = (width, height)
      resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
      img = resized
      og_img = np.array(img)
      img /= np.max(img)

      meanVal = np.mean(img)
      stdVal = np.std(img)

      # meanVal =  0.10874228924512863159
      # stdVal =  0.13627362251281738281
      img -= meanVal
      img /= stdVal
      # meanVal = np.mean(img)
      # stdVal = np.std(img)
      # img -= meanVal
      # img /= stdVal

      crop_left = img.shape[0] % (2 ** 3) // 2
      crop_right = img.shape[0] % (2 ** 3) - crop_left

      crop_up =  img.shape[1] % (2 ** 3) // 2
      crop_down =  img.shape[1] % (2 ** 3) - crop_up

      img = img[crop_left: img.shape[0] - crop_right, crop_up:img.shape[1] - crop_down]
      og_img = og_img[crop_left: og_img.shape[0] - crop_right, crop_up:og_img.shape[1] - crop_down]
      img = img.reshape(1, img.shape[0], img.shape[1], 1)

      model = get_unet(*img.shape[1:])

      model.load_weights(weights_path)

      o1, o2, out_img  = model.predict(img)
      im = out_img.reshape(img.shape[1], img.shape[2])
      im = (im-np.min(im))/(np.max(im)-np.min(im)) * 255.0
      og_img = (og_img-np.min(og_img))/(np.max(og_img)-np.min(og_img)+0.000000001) * 255.0

      path_list = img_path.split(os.sep)

      name_the_file = (path_list[-1].split('.'))[0]

      #og_img = np.power(og_img, 0.5)

      blended_im = 0.2*og_img + 0.8*im
      stacked_im = np.dstack((og_img, og_img, blended_im))
      anno_img = np.array(stacked_im).astype('uint8')
      print (anno_img.shape)
      print (og_img.shape)


      im = Image.fromarray(im).convert('L')
      og_img = Image.fromarray(og_img).convert('L')

      im.save(output_path + 'predict/' + name_the_file + '_predict.png', 'PNG')
      og_img.save(output_path + 'original/' + name_the_file + '_original.png', 'PNG')


      anno_img = Image.fromarray(anno_img).convert('RGB')
      anno_img.save(output_path + 'merged/' + name_the_file + '_annotated_mean_std_gaussian_.png', 'PNG')

  elif mode == 'ts': #time series
      # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

      directory = input_path

      dirs = os.listdir(directory)
      for idx in tqdm(range(len(dirs))):
        f = dirs[idx]
        if f[-3:] != 'tif':
            continue
        # import pdb; pdb.set_trace()
        print(f)
        fpath = os.path.join(directory, f)
        # import pdb; pdb.set_trace()
        file_prefix = os.path.splitext(f)[0]
        # import pdb; pdb.set_trace()
        out_dir = os.path.join(output_path, file_prefix)
        # out_dir = os.path.join('./_2020_kody_data/output_timeseries_anno_test', file_prefix)
        # import pdb;pdb.set_trace()
        outdir_stacked = os.path.join(out_dir, 'stacked')
        outdir_act_mask = os.path.join(out_dir, 'act_mask')
        outdir_act_org = os.path.join(out_dir, 'act_org')
        # import pdb;pdb.set_trace()
        if not os.path.exists(outdir_stacked):
            os.makedirs(outdir_stacked)

        if not os.path.exists(outdir_act_mask):
            os.makedirs(outdir_act_mask)

        if not os.path.exists(outdir_act_org):
            os.makedirs(outdir_act_org)
        # import pdb;pdb.set_trace()
        print ("Working on", file_prefix)
        with TiffFile(fpath) as tif:
          tif = tif.asarray()

          '''
          resize
          '''

          # timesteps, h,w = tif.shape
          # org_h, org_w = h, w
          # scale_percent = 140
          # # h = int(h * scale_percent / 100)
          # # h = 1120
          # # w = int(w * scale_percent / 100)
          # # w = 1120
          # dim= (h, w)
          # newTif = np.zeros((timesteps, h, w))
          # for i in range(timesteps):
          #     newTif[i,:,:] = cv2.resize(tif[i,:,:], dim, interpolation = cv2.INTER_AREA)
          #
          # tif = newTif
          '''
          end Resize
          '''
          print (tif.shape)

          num_channels = len(tif.shape)
          if num_channels == 5:
              ntimesteps = tif.shape[0]
              channel_actin = tif[:,:,0,:,:]
              channel_wall = tif[:,:,1,:,:]

              mip_actin = np.max(channel_actin, axis=1)
              _, h,w = mip_actin.shape
          elif num_channels == 4:
              ntimesteps = tif.shape[0]
              channel_actin = tif[:1,1,:,:]
              channel_wall = tif[:1,0,:,:]

              mip_actin = channel_actin
              _, h,w = mip_actin.shape
          else:
              ntimesteps = tif.shape[0]
              mip_actin = tif
              _, h,w = mip_actin.shape

          tif = mip_actin

          movie_list = []
          mt_list = []

          batch_size = 1
          tiffInputs = prediction_set_generator(tif, batch_size)

          model = get_unet(tif.shape[1], tif.shape[2], 1)
           # model.load_weights('./man/MT_parallel_128_020_030_050_Cross_hourglass_checkpoint_manual.h5')
            # model.load_weights('./' + targetPath + '/crossUp_2p5.h5') # man_2p5
            # model.load_weights('//raid1/stromules/yiliu_code/manual/_2020_kody_data/weight/at_weights.h5') # man_2p5
            # model.load_weights('//raid1/stromules/yiliu_code/manual/nate_patches_2/mean_std_one_image_crossUp_2p5_00000010.h5') # man_2p5
          model.load_weights(weights_path) # man_2p5
            # im1, im2, out_img = model.predict(mt_mip, batch_size=len(mt_mip))
          im1, im2, out_img = model.predict_generator(tiffInputs, (len(tif) // batch_size) + 1)
          out_img = np.array(out_img)
          # post processing
          for tstep in range(len(out_img)):

              curr_out_img = out_img[tstep]

              original_img = tif[tstep, :, :].astype('float32')
              original_img = np.array(original_img)
              original_img /= np.max(original_img)
              original_img = 255. * original_img


          #   mt_mip /= 255
          #   mt_mip = cv2.GaussianBlur(mt_mip, (5, 5), 0)
          #   # mt_mip = mt_mip.astype(np.uint8)
          #   # mt_mip = color.rgb2gray(mt_mip)
          #
          #   original_img = np.array(mt_mip)
              curr_out_img = curr_out_img.reshape(h, w)
              curr_out_img = (curr_out_img-np.min(curr_out_img))/(np.max(curr_out_img)-np.min(curr_out_img)) * 255.0



              og_img = original_img / np.max(original_img)
              og_img = (og_img-np.min(og_img))/(np.max(og_img)-np.min(og_img)+0.000000001) * 255.

              where_are_NaNs = np.isnan(og_img)
              og_img[where_are_NaNs] = 0
              where_are_NaNs = np.isnan(curr_out_img)
              curr_out_img[where_are_NaNs] = 0

              blended_im = 0.8*(og_img) + 0.2*curr_out_img
              stacked_im = np.dstack((blended_im, curr_out_img, curr_out_img))

              # stacked_im = cv2.resize(stacked_im, (org_h,org_w), interpolation = cv2.INTER_AREA)
              # curr_out_img = cv2.resize(curr_out_img, (org_h,org_w), interpolation = cv2.INTER_AREA)
              try:
                  io.imsave(outdir_stacked + '/' + str(tstep) + '.tif', stacked_im.astype('uint8'))
                  io.imsave(outdir_stacked + '/' + str(tstep) + '.png', stacked_im.astype('uint8'))
                  io.imsave(outdir_act_mask + '/' + str(tstep) + '.tif', curr_out_img.astype('uint8'))
                  io.imsave(outdir_act_mask + '/' + str(tstep) + '.png', curr_out_img.astype('uint8'))
                  io.imsave(outdir_act_org + '/' + str(tstep) + '.png', original_img.astype('uint8'))
              except:
                  import pdb; pdb.set_trace()
              mt_list.append(curr_out_img)
              movie_list.append(stacked_im)

          if not os.path.exists(out_dir + '/output_time_series/'):
              os.makedirs(out_dir + '/output_time_series/')
          if not os.path.exists(out_dir + '/output_time_series_movie/'):
              os.makedirs(out_dir + '/output_time_series_movie/')
          if not os.path.exists(out_dir + '/output_time_series_stacked/'):
              os.makedirs(out_dir + '/output_time_series_stacked/')

          imageio.mimsave(out_dir + '/output_time_series/' + file_prefix + '_mt.tif', mt_list)
          imageio.mimsave(out_dir + '/output_time_series_movie/' + file_prefix + '_movie.gif', movie_list, duration=0.2)
          imageio.mimsave(out_dir + '/output_time_series_stacked/' + file_prefix + '_stacked.tif', movie_list)


if __name__ == "__main__":
    print ("running")
    args = get_parser()

    main(args)

          #
          # # ntimesteps = 2
          # for timestep in range(ntimesteps):
          #   mt_mip = tif[timestep,:, :].astype('float32')
          #
          #   # import pdb;pdb.set_trace()
          #   mt_mip /= 255
          #   mt_mip = cv2.GaussianBlur(mt_mip, (5, 5), 0)
          #   # mt_mip = mt_mip.astype(np.uint8)
          #   # mt_mip = color.rgb2gray(mt_mip)
          #
          #   original_img = np.array(mt_mip)
          #   try:
          #       mt_mip -= np.mean(original_img)
          #       mt_mip /= np.std(original_img)
          #   except:
          #       import pdb; pdb.set_trace()
          #
          #   # mt_mip -= meanVal
          #   # mt_mip /= stdVal
