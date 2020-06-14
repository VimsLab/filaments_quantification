import os
import glob
import fnmatch
import numpy as np
from PIL import Image, ImageSequence

def gamma_correction(img, gamma):
  img = np.power(img, gamma)
  return img

def get_images(path, extension1,extension2, recursive):
  if not recursive:
    img_paths = glob.glob(path + extension1)
    mask_img_paths = glob.glob(path + extension2)
  else:
    img_paths = []
    for root, directories, filenames in os.walk(path):
      for filename in fnmatch.filter(filenames, extension1):
        img_paths.append(os.path.join(root,filename))
  
    mask_img_paths = []
    for root, directories, filenames in os.walk(path):
      for filename in fnmatch.filter(filenames, extension2):
        mask_img_paths.append(os.path.join(root,filename))
 
  img_paths.sort()
  mask_img_paths.sort()
  #return img_paths[0:1000], mask_img_paths[0:1000]
  return img_paths, mask_img_paths


def get_images_pre(path, extension, recursive):
  if not recursive:
    img_paths = glob.glob(path  + extension)
    img_paths = glob.glob(path  + extension)
    # img_paths = glob.glob('./Mt_Predict/' + extension)
    #mask_img_paths = glob.glob(path + '/masks/' + extension)
  else:
    img_paths = []
    #for root, directories, filenames in os.walk(path):
    #for root, directories, filenames in os.walk(path + '/originalDiFolder/'):
    #print(path)
    for root, directories, filenames in os.walk(path):
      for filename in fnmatch.filter(filenames, extension):
        img_paths.append(os.path.join(root,filename))
  
    # mask_img_paths = []
    # for root, directories, filenames in os.walk(path + '/masks/'):
    #   for filename in fnmatch.filter(filenames, extension):
    #     mask_img_paths.append(os.path.join(root,filename))
 
  img_paths.sort()

  #mask_img_paths.sort()
  print (img_paths)
  return img_paths

    
def tiffToArray (img_path_2):
    im = Image.open(img_path_2)
    gt = []
    for i, page in enumerate(ImageSequence.Iterator(im)):
        tmpPage = np.array(page)
        gt.append(tmpPage)
    gt = np.array(gt)
    # print (gt.shape)
    return gt


def get_my_patches(path1, path2, extension1, extension2, patch_size, recursive=True):
  out_size = patch_size + (1,)
  img_paths, mask_img_paths = get_images(path1, path2, extension1, extension2, recursive)
  
  train_images = []
  train_masks = []
  for ind in range(len(img_paths)):
    img = Image.open(img_paths[ind]).convert('L')
    img = np.array(img).astype('float32')

    
    mask = tiffToArray(mask_img_paths[ind]).astype('float32')
    mask = np.rollaxis(mask,0,3)

    print (ind+1,'/',len(img_paths))
    train_images.append(img.reshape(out_size))

    train_masks.append(mask)
    print ('pathces have been read')
  
  return np.array(train_images), np.array(train_masks)

