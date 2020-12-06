from matplotlib.path import Path

import scipy.stats as st
import matplotlib.patches as patches
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import json

from skimage.morphology import skeletonize
from PIL import Image, ImageDraw
from tqdm import tqdm
from get_intersection_and_endpoint import get_skeleton_endpoint, get_skeleton_intersection_and_endpoint, get_skeleton_intersection

def create_curves(width, height, num_cp, num_points, L, deform ):

    # random angle to rotate
    an_rot_st = 0
    an_rot = 180
    angle = np.deg2rad(an_rot_st -random.randint(0, an_rot))
    # angle = np.deg2rad(90)
    rot_mat = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rot_mat = np.asarray(rot_mat)

    #random translation
    trans = random.randint(0, round(width/2))
    trans_mat = [[trans*np.sin(angle)],[trans*np.cos(angle)]]

    #rand om deformations
    d =  np.random.randn(num_cp)*deform
    xp = np.linspace(-L/2,L/2,num_cp)

    pp = interpolate.splrep(xp,d);

    #generate points
    x = np.linspace(-L/2,L/2,num_points)
    y = interpolate.splev(x, pp);

    points = np.stack((x,y))
    pts = np.dot(rot_mat,points)


    pts = pts + np.tile(trans_mat,(1, num_points))

    #points to rasterize in an image
    im_pts = np.minimum( np.maximum(0,np.rint(pts+width/2 - 1)),width - 1)
    im_pts = im_pts.astype('int32')
    im_pts = tuple(map(tuple,im_pts))


    im = np.zeros((width, width))
    im[im_pts] = 1
    im[0 , :] = 0
    im[im.shape[0] - 1, :] = 0
    im[: , im.shape[1] - 1] = 0
    im[: , 0] = 0
    # plt.imshow(im)
    # plt.show()
    # import pdb; pdb.set_trace()

    return im


def main():
    gt_file = os.path.join('./dataset/', 'curves_pool_t_junction_zigzag_di_5.json')

    # directory_curve_pool = './dataset/test_pool'
    directory_curve_pool = './dataset/curves_pool_t_junction_zigzag_di_5'

    if not os.path.exists(directory_curve_pool):
        os.makedirs(directory_curve_pool)

    number_of_images = 20000
    width = 256
    height = 256

    num_cp = 10
    num_points = 10000
    L = 500
    deform = 20

    train_data = []

    for i in tqdm(range(number_of_images)):
        single_data = {}
        img_info = {}

        instances = []
        num_of_curves = random.randint(1,4)
        if random.randint(0,8) < 3:
            kernel = np.ones((3,3),np.uint8)
        else:
            kernel = np.ones((5,5),np.uint8)
        input_image = np.zeros((width, height))
        input_image_dilate = np.zeros((width, height))

        prev_skel_im_exsit = False
        all_end_point = []
        for ii in range(num_of_curves):

            instance_curve = {}

            im = create_curves(width, height, num_cp, num_points, L, deform)

            im = skeletonize(im)
            im = im * 1.


            # Make T junction
            if random.randint(0,12) < 4 and prev_skel_im_exsit != False:
                temp_map = ((prev_skel_im + im) > 0) * 1.0
                # cv2.imshow('ttt', temp_map * 255.)
                # cv2.waitKey(0)
                intersections_tmp = get_skeleton_intersection(temp_map)
                if len(intersections_tmp) != 0:

                    intersections_tmp = (intersections_tmp[0][0], intersections_tmp[0][1])
                    # import pdb;pdb.set_trace()
                    endpoints_tmp = get_skeleton_endpoint(im)
                    endpoints_tmp = (endpoints_tmp[0][0], endpoints_tmp[0][1])
                    displacement = (intersections_tmp[0] - endpoints_tmp[0],intersections_tmp[1] - endpoints_tmp[1])
                    new_im = np.roll(im, int(displacement[0]), axis = 1)
                    new_im = np.roll(new_im, int(displacement[1]), axis = 0)
                    im = new_im
                    # temp_map = ((prev_skel_im + im) > 0) * 1.0
                    # cv2.imshow('t', temp_map * 255.)
                    # cv2.waitKey(0)

            elif random.randint(0,12) >= 8 and prev_skel_im_exsit != False:
                im = create_curves(width, height, num_cp, num_points, L, deform)
                im = skeletonize(im)
                im = im * 1.

                im2 = create_curves(width, height, num_cp, num_points, L, deform)
                im2 = skeletonize(im2)
                im2 = im2 * 1.


                temp_map = ((im2 + im) > 0) * 1.0
                # cv2.imshow('ttt', temp_map * 255.)
                # cv2.waitKey(0)


                intersections_tmp = get_skeleton_intersection(temp_map)
                if len(intersections_tmp) != 0:
                    im2_skel_endpoint = get_skeleton_endpoint(im2)
                    current_skel_endpoint = get_skeleton_endpoint(im)

                    intersections_tmp = (intersections_tmp[0][0], intersections_tmp[0][1])
                    # import pdb;pdb.set_trace()
                    endpoints_tmp = get_skeleton_endpoint(im)
                    im2_skel_endpoint_A = (im2_skel_endpoint[0][0], im2_skel_endpoint[0][1])
                    current_skel_endpoint_A = (current_skel_endpoint[1][0], current_skel_endpoint[1][1])

                    displacement_im2 = (intersections_tmp[0] - im2_skel_endpoint_A[0],intersections_tmp[1] - im2_skel_endpoint_A[1])
                    displacement_curr = (intersections_tmp[0] - current_skel_endpoint_A[0],intersections_tmp[1] - current_skel_endpoint_A[1])
                    new_im2 = np.roll(im2, int(displacement_im2[0]), axis = 1)
                    new_im2= np.roll(new_im2, int(displacement_im2[1]), axis = 0)

                    new_im = np.roll(im, int(displacement_curr[0]), axis = 1)
                    new_im = np.roll(new_im, int(displacement_curr[1]), axis = 0)

                    im = ((new_im + new_im2) > 0) * 1.0
                    # temp_map = ((prev_skel_im + im) > 0) * 1.0




            skel = np.where(im > 0)
            prev_skel_im = im
            prev_skel_im_exsit = True

            endpoints = get_skeleton_endpoint(im)
            # debug = im
            for b in endpoints:
            # import pdb; pdb.set_trace()
                debug = cv2.circle(debug, (b[0], b[1]), 5, (200,0,0), 1)
                cv2.imshow('t', debug * 255.)
                cv2.waitKey(0)

            endpoints = [tuple(map(int, endpoint)) for endpoint in endpoints]
            all_end_point.extend(endpoints)
            skel = [list(map(int,skel_one_axis)) for skel_one_axis in skel]

            instance_curve['endpoints'] = endpoints
            instance_curve['skel'] =  skel
            im_dilate = cv2.dilate(im, kernel)

            input_image_dilate = input_image_dilate + im_dilate
            input_image = input_image + im
            instances.append(instance_curve)


        input_overlapping = input_image_dilate > 1
        input_overlapping = input_overlapping * 1
        # intersections = get_skeleton_intersection(input_overlapping)

        input_image = input_image > 0
        input_image = skeletonize(input_image)
        input_image = input_image * 1.0

        # intersections, _ = get_skeleton_intersection_and_endpoint(input_image)
        intersections = get_skeleton_intersection(input_image)

        intersections = [tuple(map(int, intersection)) for intersection in intersections]
        intersections.extend(all_end_point)
        # import pdb;pdb.set_trace()

        img_info ['file_name'] = str(i) + '.png'
        img_info ['file_path'] = directory_curve_pool

        single_data['instances'] = instances
        single_data['intersections'] =  intersections
        single_data ['img_info'] = img_info

        input_image = cv2.dilate(input_image, kernel)
        img_debug = Image.fromarray(input_image * 255.)
        img_debug = np.asarray(img_debug)
        # img_debug_draw = ImageDraw.Draw(img_debug)
        # import pdb; pdb.set_trace()
        ##############################################
        # for a in intersections:
        #     # import pdb; pdb.set_trace()
        #     cc = 0
        #     # img_debug = cv2.circle(img_debug, (a[0], a[1]), 3, (0,255,0), 2)
        #     # img_debug.show()
        # for a in instances:
        #     end_points = a['endpoints']
        #     for b in end_points:
        #         # import pdb; pdb.set_trace()
        #         img_debug = cv2.circle(img_debug, (b[0], b[1]), 5, (200,0,0), 1)
        # I = np.asarray(img_debug)
        # cv2.imshow('t', I)
        # cv2.waitKey(0)
        ##################################################

        train_data.append(single_data)
        input_image = Image.fromarray(input_image * 255)
        input_image.convert('L').save(directory_curve_pool + '/' + str(i) + '.png')

    print('saving transformed annotation...')
    with open(gt_file,'w') as wf:
        json.dump(train_data, wf)
        print('done')


if __name__ == '__main__':
    main()



# n = 4 # Number of possibly sharp edges
# r = .7 # magnitude of the perturbation from the unit circle,
# # should be between 0 and 1
# N = n*3+1 # number of points in the Path
# # There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

# # angles = np.linspace(0,2*np.pi,N)
# # angles = np.linspace(0,np.pi,N)

# lengths = np.linspace(0, 100, N)
# codes = np.full(N,Path.CURVE4)
# codes[0] = Path.MOVETO

# max_num_deform_points = 4
# random_deform_points_index = []

# for i in range(max_num_deform_points):
#     random_deform_points_index.append(random.randint(0, N))


# noises = st.norm.rvs(loc = 3,scale = 10,size= max_num_deform_points)
# y = np.zeros(np.size(lengths))
# import pdb; pdb.set_trace()

# for i in range(len(random_deform_points_index)):
#     idx = random_deform_points_index[i]
#     y[idx] = y[idx] + noises[i]

# import pdb; pdb.set_trace()
# verts = np.stack((lengths, y), axis=-1)

# # verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]
# # verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
# path = Path(verts, codes)

# # fig = plt.figure()
# ax = fig.add_subplot(111)
# patch = patches.PathPatch(path, facecolor='none', lw=2)
# ax.add_patch(patch)

# ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
# ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
# ax.axis('off') # removes the axis to leave only the shape

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as st
# import random
# from scipy.optimize import curve_fit

# #number of data points
# n = 50

# #function
# def func(data):
#     return 10*np.exp(-0.5*data)

# def fit(data, a, b):
#     return a*np.exp(b*data)

# #define interval
# a = 0
# b = 4

# #generate random data grid
# x = []
# for i in range(0, n):
#     x.append(random.uniform(a, b))
# x.sort()

# #noise-free data points
# yclean = []
# for i in range(0, n):
#     yclean.append(func(x[i]))

# #define mean, standard deviation, sample size for 0 noise and 1 errors
# mu0 = 0
# sigma0 = 0.4
# mu1 = 0.5
# sigma1 = 0.02

# #generate noise
# noise = st.norm.rvs(mu0, sigma0, size = n)
# y = yclean + noise
# yerr = st.norm.rvs(mu1, sigma1, size = n)

# #now x and y is your data
# #define analytic x and y
# xan = np.linspace(a, b, n)
# yan = []
# for i in range(0, n):
#     yan.append(func(xan[i]))

# #now estimate fit parameters
# #initial guesses
# x0 = [1.0, 1.0]
# #popt are list of optimal coefficients, pcov is covariation matrix
# popt, pcov = curve_fit(fit, x, y, x0, yerr)

# fity = []
# for i in range(0, n):
#     fity.append(fit(xan[i], *popt))

# print ('function used to generate is 10 * exp( -0.5 * x )')
# print ('fit function is', popt[0], '* exp(', popt[1], '* x )')

# #plotting data and analytical function
# plt.rc("figure", facecolor="w")
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif',size = 16)
# plt.title("Data", fontsize=20)
# plt.errorbar(x, y, yerr, fmt='o')
# plt.plot(xan, yan, 'r')
# plt.plot(xan, fity, 'g')
# plt.show()
