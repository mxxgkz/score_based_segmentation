""" This code comes from https://github.com/atch841/one-shot-texture-segmentation/blob/master/generate_collages.py. """
# Most of the code here are taken from https://github.com/ivust/one-shot-texture-segmentation

# %% [markdown]
## 

import numpy as np
import cv2
import os
import pickle
from PIL import Image
from joblib import Parallel, parallel_backend, delayed

def generate_texture(img_folder, ls_fnames=None):
	# Read images from dtd dataset and crop them into 256x256.
	# Images that are smaller than 256x256 are dropped.
    all_texture = []
    if ls_fnames is None:
        # Keep the order of textures
        ls_fnames = []
        for fn in list(os.listdir(img_folder)):
            if fn.endswith('.pgm'):
                ls_fnames.append(fn)
        
    for img_name in ls_fnames:
        img = cv2.imread(os.path.join(img_folder, img_name))
        if type(img) == type(None):
            print(os.path.join(img_folder, img_name), 'is not a image.')
            continue
        if img.shape[0] < 256 or img.shape[1] < 256:
            print(os.path.join(img_folder, img_name), 'is too small.')
            continue
        mid_0 = img.shape[0] // 2
        mid_1 = img.shape[1] // 2
        img = img[mid_0-128:mid_0+128,mid_1-128:mid_1+128,:]
        all_texture.append(img)
    all_texture = np.array(all_texture)
    print("The number of textures is {}, and shape of all textures is {}.".format(all_texture.shape[0], all_texture.shape))
    return all_texture

def generate_collages_batch(
        textures,
        batch_size=1,
        segmentation_regions=10,
        anchor_points=None):
    """ Generate collages using those textures. """
    N_textures = textures.shape[0]
    img_size= textures.shape[1]
    # get masks and ancher points for the batch
    masks, n_points = generate_random_masks(img_size, batch_size, segmentation_regions, anchor_points)

    # segmentation_regions * batch_size
    textures_idx = np.array([np.random.randint(0, N_textures, size=batch_size) for _ in range(segmentation_regions)])
    textures_module_idx = np.array([np.ones_like(textures[0,...,0:1])*i for i in range(N_textures)])
    batch_x = sum(textures[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    # print(masks.shape, masks[:,:,:,0:1].shape, textures_idx.shape, textures[textures_idx[0]].shape)
    # print(textures_idx[0], masks[:,:,:,0:1], textures_idx[0] * masks[:,:,:,0:1], (textures_idx[0] * masks[:,:,:,0:1]).shape)
    batch_y = sum(textures_module_idx[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    # return batch_x, batch_y, textures, textures_idx, n_points
    return batch_x, batch_y

def generate_one_collage_tf(textures,
        N_textures,
        img_size,
        segmentation_regions=10,
        n_points=None):
    # When I call this function in the map, I cannot get the shape of textures.
    # N_textures = textures.shape[0]
    # img_size= textures.shape[1]
    batch_size = 1
    # get masks and ancher points for the batch
    if n_points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
    anchor_points = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
    masks, _ = generate_random_masks(img_size, batch_size, segmentation_regions, anchor_points)

    # segmentation_regions * batch_size
    textures_idx = np.array([np.random.randint(0, N_textures, size=batch_size) for _ in range(segmentation_regions)])
    textures_module_idx = np.array([np.ones_like(textures[0,...,0:1])*i for i in range(N_textures)])
    batch_x = sum(textures[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    batch_y = sum(textures_module_idx[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    return batch_x.squeeze(axis=0), batch_y.squeeze(axis=(0,-1))

def generate_one_collage(textures,
        save_folder,
        index,
        segmentation_regions=10,
        n_points=None):
    # When I call this function in the map, I cannot get the shape of textures.
    N_textures = textures.shape[0]
    # print("The number of texture is {}.".format(N_textures))
    img_size= textures.shape[1]
    batch_size = 1
    # get masks and ancher points for the batch
    if n_points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
    anchor_points = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
    masks, _ = generate_random_masks(img_size, batch_size, segmentation_regions, anchor_points)

    # segmentation_regions * batch_size
    textures_idx = np.array([np.random.randint(0, N_textures, size=batch_size) for _ in range(segmentation_regions)])
    textures_module_idx = np.array([np.ones_like(textures[0,...,0:1])*i for i in range(N_textures)])
    batch_x = sum(textures[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    batch_y = sum(textures_module_idx[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions))
    x_arr, y_arr = batch_x.squeeze(axis=0), batch_y.squeeze(axis=(0,-1))
    if np.sum(y_arr>=N_textures):
        print("There is an ERROR. The label is larger then expected!!!")
        print(N_textures, masks.shape, textures_idx.shape, textures_module_idx.shape, 
              batch_x.shape, batch_y.shape, x_arr.shape, y_arr.shape, index, save_folder, n_points, anchor_points)
        dump_folder = os.path.expanduser(os.path.join(save_folder,'../'))
        pickle.dump(masks, open(os.path.join(dump_folder, 'masks.h5'), 'wb'))
        pickle.dump(textures_idx, open(os.path.join(dump_folder, 'textures_idx.h5'), 'wb'))
        pickle.dump(textures_module_idx, open(os.path.join(dump_folder, 'textures_module_idx.h5'), 'wb'))
        pickle.dump(batch_x, open(os.path.join(dump_folder, 'batch_x.h5'), 'wb'))
        pickle.dump(batch_y, open(os.path.join(dump_folder, 'batch_y.h5'), 'wb'))

    x_img = Image.fromarray(x_arr.astype(np.uint8), mode='RGB')
    y_img = Image.fromarray(y_arr.astype(np.uint8), mode='L')
    x_path = os.path.join(os.path.expanduser(save_folder),str(index)+'_x.png')
    y_path = os.path.join(os.path.expanduser(save_folder),str(index)+'_y.png')
    x_img.save(x_path)
    y_img.save(y_path)
    return x_path, y_path
    

def generate_random_masks(img_size=256, batch_size=1, segmentation_regions=10, points=None):
    xs, ys = np.meshgrid(np.arange(0, img_size), np.arange(0, img_size)) # Generate x, y coordinates.

    if points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
        # n_points = [segmentation_regions] * batch_size
        points   = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
    else:
        n_points = np.array([ele.shape[0] for ele in points])
        
    masks = []
    for b in range(batch_size):
        dists_b = [np.sqrt((xs - p[0])**2 + (ys - p[1])**2) for p in points[b]] # distance of every pixel to each anchor point in that batch
        voronoi = np.argmin(dists_b, axis=0) # find the index of point with the smallest distance for each pixel
        masks_b = np.zeros((img_size, img_size, segmentation_regions))
        for m in range(segmentation_regions):
            masks_b[:,:,m][voronoi == m] = 1 # The mask for texture with label m
        masks.append(masks_b)
    return np.stack(masks), n_points

def generate_validation_collages(N=2):
    textures = np.load('validation_textures.npy')
    collages = generate_collages_batch(textures, batch_size=N)
    np.save('validation_collages.npy', collages)


# %%[markdown]
# Generate training and validation images of collages

def write_txt(fname, fdir, ls_rows, ls_idx):
    f=open(os.path.join(fdir, fname),'w')
    for idx in ls_idx:
        row = ls_rows[idx]
        f.write(','.join(row)+'\n')
    f.close()

def gen_save_dataset(img_folder, sample_size, fd_name, ls_fnames=None):
    all_textures = generate_texture(img_folder, ls_fnames=ls_fnames)
    num_classes = all_textures.shape[0]
    img_size = all_textures.shape[1]
    save_folder = os.path.join(img_folder, fd_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    n_points = np.random.randint(2, num_classes+1, size=(sample_size,1))
    ls_tasks = [(all_textures, save_folder, idx, num_classes, n_p) for idx, n_p in enumerate(n_points)]
    with parallel_backend('loky', n_jobs=10):
        ls_img_path = Parallel(verbose=10, pre_dispatch="1.2*n_jobs")(delayed(generate_one_collage)(*task) for task in ls_tasks)
    write_txt(fd_name+'.txt', img_folder, ls_img_path, list(range(len(ls_img_path))))



def gen_save_train_valid_test_dataset(img_folder, train_size, valid_size, test_size, ls_fnames=None):
    gen_save_dataset(img_folder, train_size, 'train', ls_fnames=ls_fnames)
    gen_save_dataset(img_folder, valid_size, 'valid', ls_fnames=ls_fnames)
    gen_save_dataset(img_folder, test_size, 'test', ls_fnames=ls_fnames)
    
# img_folder = os.path.expanduser('~/scratch/Data/texture/Brodatz/5_texture_images_5c')
# train_size, valid_size, test_size = 20000, 2000, 5000
# gen_save_train_valid_test_dataset(img_folder, train_size, valid_size, test_size)


# # %% [markdown]
# ## Generate collages images
# import matplotlib.pyplot as plt

# # image_folder = os.path.expanduser('~/scratch/Data/texture/Brodatz/5_texture_images')
# image_folder = '/projects/p30309/Data/texture/Brodatz/5_texture_images'
# all_textures = generate_texture(image_folder)
# batch_x, batch_y, textures, textures_idx, n_points = generate_collages(all_textures,
#                                     batch_size=2,
#                                     segmentation_regions=10,
#                                     anchor_points=None)


# # The following code aims to confirm the correctness of the collages and 
# # labels.

# # %%
# plt.imshow(batch_x[0].astype(np.uint8))

# # %%
# plt.imshow(batch_y[0].squeeze(axis=-1), cmap='gray')
# plt.colorbar()

# # %%
# plt.imshow(batch_x[1].astype(np.uint8))

# # %%
# plt.imshow(batch_y[1].squeeze(axis=-1), cmap='gray')

# # %%
# textures_idx

# # %%
# n_points

# # %%
# plt.imshow(textures[0].astype(np.uint8))

# # %%
# plt.imshow(textures[9].astype(np.uint8))

# # %%
# plt.imshow(textures[20].astype(np.uint8))

# # %%
# plt.imshow(textures[22].astype(np.uint8))

# # %%
