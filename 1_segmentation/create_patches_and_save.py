import os
import glob
import fnmatch
import numpy as np
import cv2
from skimage import io, color
from skimage.measure import label
from skimage.transform import rotate
from skimage.util import view_as_windows
from PIL import Image
import sys



def more_patches(path, rotation_angles, gamma_correction_p, down_scale, extension, recursive):

    patch_size = (256,256)
    #out_size = patch_size + (1,)
    img_paths, mask_img_paths = get_images('.', extension, recursive)

    sum_patch = np.zeros(patch_size) # to get mean and std

    train_images = []
    train_masks = []

    count = 0
    for ind in range(len(mask_img_paths)):
        # # img = np.asarray(Image.open(img_paths[ind]).convert('L')).astype('float32')
        # img = np.asarray(Image.open(img_paths[ind]))
        # import pdb;pdb.set_trace()
        # mask = np.asarray(Image.open(mask_img_paths[ind]).convert('L')).astype('float32')
        # import pdb;pdb.set_trace()

        # cv2.imshow('mask', mask * 255.)
        # cv2.imshow('img', img[:,:,:3] * 255.)
        # cv2.waitKey(0)
        img = io.imread(img_paths[ind]).astype('float32')
        mask = io.imread(mask_img_paths[ind]).astype('float32')

        img /= np.max(img)
        img = color.rgb2gray(img)
        # import pdb;pdb.set_trace()


        # kernel = np.ones((5,5),np.float32)
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        # denoised_gray = cv2.fastNlMeansDenoising(img, None, 9, 13)


        mask = mask / 255.
        mask = color.rgb2gray(mask)

        # cv2.imshow('mask', mask )
        # cv2.imshow('img', img * 255 )
        # cv2.waitKey(0)

        org_img = img
        org_mask = mask
        #erode the orginal mask
        kernel = np.ones((3,3),np.float32)
        org_mask = cv2.erode(org_mask,kernel,iterations = 1)
    ####################
        for scale in np.nditer(down_scale):
            img = cv2.resize(org_img, None, fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_LANCZOS4 )

            mask = cv2.resize(org_mask, None, fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_AREA )

            print (img.shape)
            print (ind+1,'/',len(img_paths))
            try:
                img_patches = view_as_windows(img, patch_size, step=int(patch_size[0]/4))
            except:
                import pdb; pdb.set_trace()
            mask_patches = view_as_windows(mask, patch_size, step=int(patch_size[0]/4))


            for patch_ind_0 in range(img_patches.shape[0]):
                for patch_ind_1 in range(img_patches.shape[1]):
                    mask_patch = mask_patches[patch_ind_0,patch_ind_1]
                    if (mask_patch > 0.5).sum() > 200:# prev 70
                        img_patch = img_patches[patch_ind_0,patch_ind_1]


                        #print (np.min(img_patch))
                        # if count >= 1000:
                        #     sys.exit()
                        for angle in np.nditer(rotation_angles):
                            for gamma in np.nditer(gamma_correction_p):
                                count += 1
                                #print ("sequence? " + str(count))
                                # import pdb;pdb.set_trace()
                                one_image_patch = gamma_correction(rotate(img_patch, angle), gamma).astype('float32')
                                one_mask_patch = rotate(mask_patch, angle).astype('float32')
                                # mask = cv2.resize(one_mask_patch, (512,512), interpolation = cv2.INTER_LINEAR)
                                # cv2.imshow('t',mask)
                                # cv2.waitKey(0)
                                # kernel = np.ones((3,3))
                                # one_mask_patch = cv2.erode(one_mask_patch,kernel,iterations = 1)
                                # mask = cv2.resize(one_mask_patch, (512,512), interpolation = cv2.INTER_LINEAR)
                                # cv2.imshow('tt',mask)
                                # cv2.waitKey(0)

                                #print one_mask_patch
                                #print (one_mask_patch, count)
                                #print count
                                #cv2.putText(one_image_patch, text= str(count) + 'IAMGE', org=(5,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=c, thickness=1.5)
                                #cv2.putText(one_mask_patch, text= str(count) + 'MASK', org=(5,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=c, thickness=1.5)

                                #sum_patch += one_image_patch;
                                #print(sum_patch)
                                #print (count)
                            #For get std and mean
                                train_images.append(one_image_patch.astype('float32'))
                                # cv2.imshow('denoise', one_image_patch)
                                # cv2.waitKey()
                                cv2.imwrite(img_dir + '/img_patch_' + str(count) + '.png', one_image_patch * 255.)
                                cv2.imwrite(mask_dir + '/mask_patch_' + str(count) + '.png', one_mask_patch * 255.)
                                #cv2.imwrite('sample/img_list/img_patch_' + str(count) + '.png', one_image_patch * 255.)
                                #cv2.imwrite('sample/mask_list/mask_patch_' + str(count) + '.png', one_mask_patch * 255.)
                        for gamma in np.nditer(gamma_correction_p):
                            count += 1
                            one_image_patch = gamma_correction(np.fliplr(img_patch), gamma).astype('float32')
                            one_mask_patch = np.fliplr(mask_patch).astype('float32')

                            train_images.append(one_image_patch.astype('float32'))
                            cv2.imwrite(img_dir + '/img_patch_' + str(count) + '.png', one_image_patch * 255.)
                            cv2.imwrite(mask_dir + '/mask_patch_' + str(count) + '.png', one_mask_patch * 255.)
                        for gamma in np.nditer(gamma_correction_p):
                            count += 1

                            one_image_patch = gamma_correction(np.flipud(img_patch), gamma).astype('float32')
                            one_mask_patch = np.flipud(mask_patch).astype('float32')

                            train_images.append(one_image_patch.astype('float32'))
                            cv2.imwrite(img_dir + '/img_patch_' + str(count) + '.png', one_image_patch * 255.)
                            cv2.imwrite(mask_dir + '/mask_patch_' + str(count) + '.png', one_mask_patch * 255.)
                        print ("num_of_patches is %d              \r" %count) ,

    train_images = np.array(train_images)
    # #print(train_images.shape)
    meanVal = np.mean(train_images)
    stdVal = np.std(train_images)
    # print ('Mean: ' + str(meanVal))
    # print ('Std: ' + str(stdVal))
    # #mean = sum_patch / count;
    # #print (mean)
    f= open(fileName + "/mean_std.txt","w+")
    f.write("#mean:\n %.20f \n#std:\n %.20f\n" % (meanVal, stdVal))
    print (count)

    return



def gamma_correction(img, gamma):
    #img should be [0, 1]
    img = np.power(img, gamma)
    #print (img)
    return img

def get_images(path, extension, recursive):
    image_path = path + '/images'
    mask_path = path + '/masks'
    img_paths = []
    mask_img_paths = []


    for root, directories, filenames in os.walk(image_path):
      for filename in fnmatch.filter(filenames, extension1):
        img_paths.append(os.path.join(root,filename))

    for root, directories, filenames in os.walk(mask_path):
      for filename in fnmatch.filter(filenames, extension2):
        mask_img_paths.append(os.path.join(root,filename))

    img_paths.sort()
    mask_img_paths.sort()

    return img_paths, mask_img_paths

fileName = sys.argv[1]

img_dir =  fileName + '/img_list'
mask_dir = fileName + '/mask_list'

if not os.path.exists(fileName):
    os.makedirs(fileName)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
    os.makedirs(mask_dir)


rotation_angles = np.array([0, 90, 180, 270])
gamma_correction_p = np.array([1,1.2,0.8])
# gamma_correction_p = np.array([1])
down_scale = np.array([1.0, 1.0])


more_patches('./' + fileName, rotation_angles, gamma_correction_p, down_scale, extension='*.png', recursive = True)
