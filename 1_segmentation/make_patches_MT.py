import os
import glob
import fnmatch
import numpy as np

def gamma_correction(img, gamma):
  img = np.power(img, gamma)
  return img

def get_images(path, extension1,extension2, recursive):
  if not recursive:
    img_paths = glob.glob(path + '/originalDiFolder/' + extension1)
    mask_img_paths = glob.glob(path + '/groundTruth/' + extension2)
  else:
    img_paths = []
    for root, directories, filenames in os.walk(path + '/originalDiFolder/'):
      for filename in fnmatch.filter(filenames, extension1):
        img_paths.append(os.path.join(root,filename))
  
    mask_img_paths = []
    for root, directories, filenames in os.walk(path + '/groundTruth/'):
      for filename in fnmatch.filter(filenames, extension2):
        mask_img_paths.append(os.path.join(root,filename))
 
  img_paths.sort()
  mask_img_paths.sort()
  print (img_paths, mask_img_paths)
  return img_paths[0:100], mask_img_paths[0:100]
  # return img_paths, mask_img_paths

def get_images_pre(path, extension, recursive):
  if not recursive:
    img_paths = glob.glob(path + extension)
    #mask_img_paths = glob.glob(path + '/masks/' + extension)
  else:
    img_paths = []
    for root, directories, filenames in os.walk(path):
      for filename in fnmatch.filter(filenames, extension):
        img_paths.append(os.path.join(root,filename))
  
  img_paths.sort()

  print (img_paths)
  return img_paths

    
def tiffToArray (img_path_2):
    im = Image.open(img_path_2)
    gt = []
    for i, page in enumerate(ImageSequence.Iterator(im)):
        tmpPage = np.asarray(page)
        gt.append(tmpPage)
    gt = np.asarray(gt)
    # print (gt.shape)
    return gt


def get_my_patches(path, extension='*.png', recursive=True):
  patch_size = (64,64)
  out_size = patch_size + (1,)
  img_paths, mask_img_paths = get_images(path, '*.png', '*.tiff', recursive)
  
  train_images = []
  train_masks = []
  for ind in range(len(img_paths)):
    img = io.imread(img_paths[ind]) .astype('float32')
    print (img.max(0).max(0))
    #print (img)
    # img /= 65535.
    # img = color.rgb2gray(img)
    
    mask = tiffToArray(mask_img_paths[ind]).astype('float32')
    mask = np.rollaxis(mask,0,3)

    
    print (ind+1,'/',len(img_paths))
    train_images.append(img.reshape(out_size))

    train_masks.append(mask)
    print("Here is the problem?")
  
  return np.array(train_images), np.array(train_masks)

