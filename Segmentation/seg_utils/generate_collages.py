""" This code comes from https://github.com/atch841/one-shot-texture-segmentation/blob/master/generate_collages.py. """
# Most of the code here are taken from https://github.com/ivust/one-shot-texture-segmentation

# %% [markdown]
## 

import numpy as np
# import cv2
import os
import pickle
from PIL import Image
from joblib import Parallel, parallel_backend, delayed
from constants import *

def generate_texture(img_folder, ls_fnames=None, new_size=256, trfm_flag=False):
	# Read images from dtd dataset and crop them into 256x256.
	# Images that are smaller than 256x256 are dropped.
    all_texture = []
    if ls_fnames is None:
        # Keep the order of textures
        ls_fnames = []
        for fn in list(os.listdir(img_folder)):
            if fn.endswith('.png'):
                ls_fnames.append(fn)

    for img_name in ls_fnames:
        img = Image.open(os.path.join(img_folder, img_name))
        img = img.convert(mode='RGB')
        # img = cv2.imread(os.path.join(img_folder, img_name)) # The dimension is [height, width, channel]
        if type(img) == type(None):
            print(os.path.join(img_folder, img_name), 'is not a image.')
            continue
        if img.size[0] < 256 or img.size[1] < 256:
            print(os.path.join(img_folder, img_name), 'is too small.')
            continue
        if trfm_flag:
            # Scale
            scale_ratio = np.random.uniform(low=1.0, high=1+MAX_SCALE_RATIO)
            tmp_size = int(scale_ratio*img.size[0])
            # cv2.resize(img, (tmp_size, tmp_size), interpolation=cv2.INTER_LINEAR)
            img = img.resize((tmp_size, tmp_size), resample=Image.BILINEAR)
            # Rotate
            img = img.rotate(angle=np.random.randint(low=0, high=4)*90, resample=Image.BILINEAR)
            # Flip
            if np.random.randint(low=0, high=2):
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        mid_0 = img.size[0] // 2
        mid_1 = img.size[1] // 2
        img = img.crop((mid_1-128, mid_0-128, mid_1+128, mid_0+128))
        # img = img[mid_0-128:mid_0+128,mid_1-128:mid_1+128,:]
        if new_size!=256:
            img = img.resize((new_size, new_size), resample=Image.BILINEAR)
            # img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        all_texture.append(np.asarray(img))
    all_texture = np.array(all_texture)
    print("The number of textures is {}, and shape of all textures is {}.".format(all_texture.shape[0], all_texture.shape))
    return all_texture

def generate_collages_batch(
        textures,
        batch_size=1,
        segmentation_regions=10,
        anchor_points=None,
        pwei_flag=False,
        normp=2):
    """ Generate collages using those textures. """
    N_textures = textures.shape[0]
    img_size= textures.shape[1]
    # get masks and ancher points for the batch
    masks, n_points = generate_random_masks(img_size, batch_size, segmentation_regions, points=anchor_points, pwei_flag=pwei_flag, normp=normp)

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
        n_points=None,
        pwei_flag=False,
        normp=2):
    # When I call this function in the map, I cannot get the shape of textures.
    # N_textures = textures.shape[0]
    # img_size= textures.shape[1]
    batch_size = 1
    # get masks and ancher points for the batch
    if n_points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
    anchor_points = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
    masks, _ = generate_random_masks(img_size, batch_size, segmentation_regions, points=anchor_points, pwei_flag=pwei_flag, normp=normp)

    # segmentation_regions * batch_size
    textures_idx = np.array([np.random.randint(0, N_textures, size=batch_size) for _ in range(segmentation_regions)])
    textures_module_idx = np.array([np.ones_like(textures[0,...,0:1])*i for i in range(N_textures)])
    batch_x = sum(textures[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    batch_y = sum(textures_module_idx[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    return batch_x.squeeze(axis=0), batch_y.squeeze(axis=(0,-1))

def generate_one_collage(
        textures,
        save_folder,
        index,
        segmentation_regions=10,
        n_points=None,
        pwei_flag=False,
        trfm_flag=False,
        normp=2,
        nb=2,
        nrot=0):
    # When I call this function in the map, I cannot get the shape of textures.
    N_textures = textures.shape[0]
    # print("The number of texture is {}.".format(N_textures))
    img_size= textures.shape[1]
    batch_size = 1
    # get masks and ancher points for the batch
    if n_points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
    anchor_points = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
    masks, _ = generate_random_masks(img_size, batch_size, segmentation_regions, points=anchor_points, pwei_flag=pwei_flag, normp=normp)

    # segmentation_regions * batch_size
    textures_idx = np.array([np.random.randint(0, N_textures, size=batch_size) for _ in range(segmentation_regions)])
    textures_module_idx = np.array([np.ones_like(textures[0,...,0:1])*i for i in range(N_textures)]) # This way we can differentiate label 0 and boundary pixels.

    def trfm_textures(arr_textures, arr_rotate, arr_flip):
        if arr_textures.shape[0] != arr_rotate.shape[0] or arr_textures.shape[0] != arr_flip.shape[0]:
            raise ValueError('The texture({}), rotate({}), and flip({}) should have the same first element of shapes.'.format(arr_textures.shape, arr_rotate.shape, arr_flip.shape))
        for i, (tex, rot, flip) in enumerate(zip(arr_textures, arr_rotate, arr_flip)):
            arr_textures[i] = np.rot90(tex, k=rot)
            if flip:
                arr_textures[i] = np.fliplr(arr_textures[i])
        return arr_textures

    
    def im2col_sliding_strided(A, BSZ, stepsize=1):
        # https://stackoverflow.com/a/30110497/4307919
        # Parameters
        m,n = A.shape
        s0, s1 = A.strides    
        nrows = m-BSZ[0]+1
        ncols = n-BSZ[1]+1
        shp = BSZ[0],BSZ[1],nrows,ncols
        strd = s0,s1,s0,s1

        out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
        return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

    def mark_boundary(arr, val, nb=2):
        # 0 is background. Positive value is foreground
        mask = 1*(arr!=0)
        arr_padded = np.pad(arr, pad_width=nb, mode='reflect') # Add boundary reflectively
        im2col = im2col_sliding_strided(arr_padded, (2*nb+1,2*nb+1)) # Similar to the operation of image to convolutional matrix
        # bd_idx = np.where(np.sum(im2col!=im2col[2*nb*(nb+1)], axis=0)!=0)[0] # Find those index that it is not uniform within its broundary
        bd_idx = np.where(np.sum(im2col==0, axis=0)!=0)[0] # Easier way to find boundary, just to see if have 0 in neighbors.
        arr[bd_idx//arr.shape[1], bd_idx%arr.shape[1]] = val # Modified the original array
        return arr*mask # Get rid of those marks in the background.

    def mark_boundary_batch(batch_arr, val, nb=2):
        last_dim_1 = False
        if batch_arr.shape[-1] == 1:
            last_dim_1 = True
            batch_arr = batch_arr.squeeze(axis=-1)
        for idx, arr in enumerate(batch_arr):
            batch_arr[idx] = mark_boundary(arr, val, nb=nb)
        return batch_arr if not last_dim_1 else np.expand_dims(batch_arr, axis=-1) # Keep the original dimension in the input.

    # Make transformation for each segmentation patch.
    if trfm_flag:
        arr_rotate = np.random.randint(low=0, high=4, size=(segmentation_regions, batch_size))
        arr_flip = np.random.randint(low=0, high=2, size=(segmentation_regions, batch_size))
        batch_x = sum(trfm_textures(textures[textures_idx[i]], arr_rotate[i], arr_flip[i]) * masks[:,:,:,i:i+1] for i in range(segmentation_regions)) 
    else:
        batch_x = sum(textures[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions))
    batch_y = sum(textures_module_idx[textures_idx[i]] * masks[:,:,:,i:i+1] for i in range(segmentation_regions))
    batch_y_bd = sum(
        mark_boundary_batch((textures_module_idx[textures_idx[i]]+1) * masks[:,:,:,i:i+1], 
                            val=N_textures+1, nb=nb)  for i in range(segmentation_regions))-1 # This way we can differentiate label 0 and boundary pixels.
    x_arr, y_arr, y_bd_arr = batch_x.squeeze(axis=0).astype(np.uint8), batch_y.squeeze(axis=(0,-1)).astype(np.uint8), batch_y_bd.squeeze(axis=(0,-1)).astype(np.uint8)
    
    if np.sum(y_arr>=N_textures):
        raise ValueError("There is an ERROR. The number of label is larger then expected({})!!!".format(N_textures))
        print(N_textures, masks.shape, textures_idx.shape, textures_module_idx.shape, 
              batch_x.shape, batch_y.shape, x_arr.shape, y_arr.shape, index, save_folder, n_points, anchor_points)
        dump_folder = os.path.expanduser(os.path.join(save_folder,'../'))
        pickle.dump(masks, open(os.path.join(dump_folder, 'masks.h5'), 'wb'))
        pickle.dump(textures_idx, open(os.path.join(dump_folder, 'textures_idx.h5'), 'wb'))
        pickle.dump(textures_module_idx, open(os.path.join(dump_folder, 'textures_module_idx.h5'), 'wb'))
        pickle.dump(batch_x, open(os.path.join(dump_folder, 'batch_x.h5'), 'wb'))
        pickle.dump(batch_y, open(os.path.join(dump_folder, 'batch_y.h5'), 'wb'))

    # None 90-degree rotation
    for _ in range(nrot):
        # Center and angle
        center = tuple(np.random.randint(low=0, high=img_size, size=2))
        angle = np.random.uniform()*360
        # print(center, angle)
        # Add 1 to label so that we can differentiate labels and 0 as background
        y_arr += 1
        y_bd_arr += 1
        # Get image for ratation
        # print(x_arr.shape, x_arr.dtype, x_arr)
        x_img = Image.fromarray(x_arr) # pillow can generate image from (256,256,3) array
        y_img = Image.fromarray(y_arr) # pillow can generate image from (256,256) array
        y_bd_img = Image.fromarray(y_bd_arr)
        x_img_rot = x_img.rotate(angle, resample=Image.BILINEAR, expand=False, center=center)
        ## Notice that for label we cannot use Image.BILINEAR. OW, there will some new labels falsely created.
        y_img_rot = y_img.rotate(angle, resample=Image.NEAREST, expand=False, center=center)
        y_bd_img_rot = y_bd_img.rotate(angle, resample=Image.NEAREST, expand=False, center=center)
        # print(np.unique(y_img_rot), np.unique(y_img))
        # x_img
        # x_img_rot
        # labs_rot, labs = np.unique(y_img_rot)[1:]-1, np.unique(y_img)-1
        # if not np.all(labs_rot==labs):
        #     raise ValueError("The image before ({}({})) and after ({}({})) rotation should have the same labels.".format(labs, labs.shape[0], labs_rot, labs_rot.shape[0]))
        # Construct image with rotated and background randomly chosen from textures.
        rot_mask = np.expand_dims(1*(np.array(y_img_rot) != 0), axis=-1) # Rotation automatically fill 0 in background
        bg_idx = np.random.randint(low=0, high=N_textures)
        bg_texture = textures[bg_idx]
        x_arr = np.array(x_img_rot)*rot_mask + bg_texture*(1-rot_mask)
        y_arr = np.array(y_img_rot) + (bg_idx+1)*(1-rot_mask.squeeze(axis=-1)) - 1 # Prevent overflow for np.uint8.
        y_bd_arr_comp = np.array([np.array(y_bd_img_rot), (bg_idx+1)*(1-rot_mask.squeeze(axis=-1))])
        y_bd_arr = np.sum(mark_boundary_batch(y_bd_arr_comp, val=N_textures+1, nb=nb), axis=0) - 1
        x_arr, y_arr, y_bd_arr = x_arr.astype(dtype=np.uint8), y_arr.astype(dtype=np.uint8), y_bd_arr.astype(dtype=np.uint8)

    x_img = Image.fromarray(x_arr, mode='RGB')
    y_img = Image.fromarray(y_arr, mode='L')
    y_bd_img = Image.fromarray(y_bd_arr, mode='L')
    x_path = os.path.join(os.path.expanduser(save_folder),str(index)+'_x.png')
    y_path = os.path.join(os.path.expanduser(save_folder),str(index)+'_y.png')
    y_bd_path = os.path.join(os.path.expanduser(save_folder),str(index)+'_y_bd.png')
    x_img.save(x_path)
    y_img.save(y_path)
    y_bd_img.save(y_bd_path)
    return x_path, y_path, y_bd_path

def generate_random_masks(img_size=256, batch_size=1, segmentation_regions=10, points=None, pwei_flag=False, normp=2, weights=None):
    """
    Add weights for different anchor points so that the boundary can be curved line.

    Return:
        arr_masks: A array of masks so that the first dimension is the batch size.
        n_points: The number of points for each image in the batch for segmenting those images. Has the same dimension as the first dimension of arr_masks.
    """
    def dist_pnorm(xs, ys, pt, normp):
        return (np.abs(xs-pt[0])**normp+np.abs(ys-pt[1])**normp)**(1/normp)

    xs, ys = np.meshgrid(np.arange(0, img_size), np.arange(0, img_size)) # Generate x, y coordinates.

    if points is None:
        n_points = np.random.randint(2, segmentation_regions + 1, size=batch_size)
        # n_points = [segmentation_regions] * batch_size
        points   = [np.random.randint(0, img_size, size=(n_points[i], 2)) for i in range(batch_size)]
    else:
        n_points = np.array([ele.shape[0] for ele in points])

    if weights is None and pwei_flag: # The weights for each centroids
        weights = [np.random.uniform(0.5, 1.5, size=(n_points[i], )) for i in range(batch_size)]
    elif weights is None:
        weights = [np.ones((n_points[i], )) for i in range(batch_size)]
        
    masks = []

    if normp<=0:
        normp = np.random.uniform(low=0.5, high=3.5)

    for b in range(batch_size):
        dists_b = [w*dist_pnorm(xs, ys, pt, normp) for pt, w in zip(points[b], weights[b])] # distance of every pixel to each anchor point in that batch; each anchor point has a weight.
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

def write_txt(fname, fdir, ls_rows, ls_idx, foption='w'):
    f=open(os.path.join(fdir, fname), foption)
    for idx in ls_idx:
        row = ls_rows[idx]
        f.write(','.join(row)+'\n')
    f.close()

def shuffle_lines_txt(fname, fdir):
    with open(os.path.join(fdir, fname), 'r') as readf:
        lines = readf.readlines()
    rand_lidx = np.arange(len(lines))
    np.random.shuffle(rand_lidx)
    
    with open(os.path.join(fdir, fname), 'w') as writef:
        for lidx in rand_lidx:
            row = lines[lidx]
            writef.write(row)

def gen_save_dataset(img_folder, sample_size, fd_name, ls_fnames=None, new_size=256, pwei_flag=False, normp=2, trfm_flag=False, num_gen_batch=1, nb=2, max_rots=3):
    sample_size = sample_size if sample_size%num_gen_batch==0 else (sample_size//num_gen_batch+1)*num_gen_batch
    shuffled_idx = np.random.choice(np.arange(sample_size), size=sample_size, replace=False)
    chunk_size = shuffled_idx.shape[0]//num_gen_batch
    ls_shuffled_idx = [shuffled_idx[i*chunk_size:(i+1)*chunk_size] for i in range(num_gen_batch)]
    for i in range(num_gen_batch):
        all_textures = generate_texture(img_folder, ls_fnames=ls_fnames, new_size=new_size, trfm_flag=trfm_flag)
        print("The textures shape are {}.".format(all_textures.shape))
        num_classes = all_textures.shape[0]
        img_size = all_textures.shape[1]
        save_folder = os.path.join(img_folder, fd_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        n_points = np.random.randint(2, num_classes+1, size=(chunk_size,1))
        n_rots = np.random.randint(0, max_rots+1, size=chunk_size)
        ls_tasks = [(all_textures, save_folder, idx, num_classes, n_p, pwei_flag, trfm_flag, normp, nb, n_r) for idx, n_p, n_r in zip(ls_shuffled_idx[i], n_points, n_rots)]
        with parallel_backend('loky', n_jobs=PIPLINE_JOBS):
            ls_img_path = Parallel(verbose=2, pre_dispatch="2*n_jobs")(delayed(generate_one_collage)(*task) for task in ls_tasks)
        write_txt(fd_name+'.txt', img_folder, ls_img_path, list(range(len(ls_img_path))), foption='w' if i==0 else 'a')

def gen_save_train_valid_test_dataset(img_folder, train_size, valid_size, test_size, ls_fnames=None, new_size=256, pwei_flag=False, normp=2, trfm_flag=False, num_gen_batch=1, nb=2, max_rots=3):
    gen_save_dataset(img_folder, train_size, 'train', ls_fnames=ls_fnames, new_size=new_size, pwei_flag=pwei_flag, normp=normp, trfm_flag=trfm_flag, num_gen_batch=num_gen_batch, nb=nb, max_rots=max_rots)
    gen_save_dataset(img_folder, valid_size, 'valid', ls_fnames=ls_fnames, new_size=new_size, pwei_flag=pwei_flag, normp=normp, trfm_flag=trfm_flag, num_gen_batch=num_gen_batch, nb=nb, max_rots=max_rots)
    gen_save_dataset(img_folder, test_size, 'test', ls_fnames=ls_fnames, new_size=new_size, pwei_flag=pwei_flag, normp=normp, trfm_flag=trfm_flag, num_gen_batch=num_gen_batch, nb=nb, max_rots=max_rots)
    
# %%
# Generate image patches from a big image
def generate_image_patches(img_folder, img_lab_folder, save_folder, fd_name, lab_fname_func, targ_labs=None, stride=10, patch_size=IMG_SIZE, num_workers=PIPLINE_JOBS):
    if os.path.exists(os.path.join(save_folder, fd_name+'.txt')):
        # Remove the text file of file paths, if it already exists.
        os.remove(os.path.join(save_folder, fd_name+'.txt'))
    
    accu_idx = 0

    def paral_helper(ch_ri, ch_ci, img_arr, img_lab_arr, patch_size, save_subfolder, targ_labs, ch_idx, accu_idx):        
        ls_img_paths = []
        for idx, ri, ci in zip(ch_idx, ch_ri, ch_ci):
            img_pat_arr, img_lab_pat_arr = img_arr[ri:ri+patch_size, ci:ci+patch_size, ...].astype(np.uint8), img_lab_arr[ri:ri+patch_size, ci:ci+patch_size, ...].astype(np.uint8)
            x_path, y_path = os.path.join(save_subfolder, '{}_x.png'.format(accu_idx+idx)), os.path.join(save_subfolder, '{}_y.png'.format(accu_idx+idx))
            Image.fromarray(img_pat_arr, mode='RGB').save(x_path)
            Image.fromarray(img_lab_pat_arr, mode='L').save(y_path)
            if targ_labs is not None:
                if np.intersect1d(np.unique(img_lab_pat_arr), targ_labs).shape[0]>0:
                    # Only store those images has matrix and inclusion, excluding edge background.
                    ls_img_paths.append([x_path, y_path])
            else:
                ls_img_paths.append([x_path, y_path])
        return ls_img_paths

    # Calculate class weights
    lab_wei_map = {}
    tot_cnt = 0

    for fidx, fn in enumerate(list(os.listdir(img_folder))):
        print("Processing file {} of file name {}.".format(fidx, fn))
        if not fn.endswith('.png'):
            continue
        # fn_lab = fn.replace('.png', '_L.png')
        fn_lab = lab_fname_func(fn)
        img = Image.open(os.path.join(img_folder, fn))
        print("The image has size {}.".format(img.size))
        img_lab = Image.open(os.path.join(img_lab_folder, fn_lab))
        img_arr, img_lab_arr = np.array(img), np.array(img_lab)
        # pad around the boundary using mirror condition, so that we have more data
        # We don't want to padding validation and testing data sets
        if fd_name == 'train':
            if len(img_arr.shape)==2:
                # pad_width = patch_size
                pad_width = ((patch_size,patch_size),(patch_size,patch_size))
            else: # len(img_arr.shape)==3
                pad_width = ((patch_size,patch_size),(patch_size,patch_size),(0,0))
        else:
            pad_width = 0
        ext_img_arr = np.pad(img_arr, pad_width=pad_width, mode='reflect') # Use the boundary pixel as reflective axis. 
        ext_img_lab_arr = np.pad(img_lab_arr, pad_width=pad_width[:2] if fd_name=='train' else 0, mode='reflect')
        
        uni_labs, uni_cnts = np.unique(ext_img_lab_arr, return_counts=True)
        for lab, cnt in zip(uni_labs, uni_cnts):
            if lab in lab_wei_map:
                lab_wei_map[lab] += cnt
            else:
                lab_wei_map[lab] = cnt
            tot_cnt += cnt

        ext_img_h, ext_img_w = ext_img_arr.shape[0], ext_img_arr.shape[1]
        print("The extended image and its label has shape {}, {}.".format(ext_img_arr.shape, ext_img_lab_arr.shape))
        lc_ci, lc_ri = np.meshgrid(np.arange(0, ext_img_w-patch_size+1, stride), np.arange(0, ext_img_h-patch_size+1, stride))
        lc_ci, lc_ri = lc_ci.ravel(order='C'), lc_ri.ravel(order='C')
        save_subfolder = os.path.join(save_folder, fd_name)
        if not os.path.exists(save_subfolder):
            os.makedirs(save_subfolder)

        chunk_size = lc_ci.shape[0]//num_workers if lc_ci.shape[0]%num_workers==0 else lc_ci.shape[0]//num_workers+1
        ls_idx = list(range(lc_ci.shape[0]))
        ls_chunck_ci = [lc_ci[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers)]
        ls_chunck_ri = [lc_ri[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers)]
        ls_chunck_idx = [ls_idx[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers)]

        ls_tasks = [(ch_ri, ch_ci, ext_img_arr, ext_img_lab_arr, patch_size, save_subfolder, targ_labs, ch_idx, accu_idx) for ch_ci, ch_ri, ch_idx in zip(ls_chunck_ci, ls_chunck_ri, ls_chunck_idx)]
        with parallel_backend('loky', n_jobs=PIPLINE_JOBS):
            ls_path_res = Parallel(verbose=5, pre_dispatch="2*n_jobs")(delayed(paral_helper)(*task) for task in ls_tasks)
        for ch_paths in ls_path_res:
            write_txt(fd_name+'.txt', save_folder, ch_paths, list(range(len(ch_paths))), foption='a')
        
        accu_idx += lc_ci.shape[0]

    # Shuffle the order of all saved image patches
    shuffle_lines_txt(fd_name+'.txt', save_folder)

    print("The total number of pixels: {}".format(tot_cnt))
    print("The label frequency are: {}".format(lab_wei_map))
    for key, val in lab_wei_map.items():
        lab_wei_map[key] = tot_cnt/val/len(lab_wei_map)

    print("The label weight are: {}".format(lab_wei_map))
    return lab_wei_map
        

def gen_save_train_valid_test_patch_dataset(
    tr_img_folder, tr_img_lab_folder,
    val_img_folder, val_img_lab_folder,
    te_img_folder, te_img_lab_folder,
    save_folder, lab_fname_func, targ_labs=None, stride=10, 
    patch_size=IMG_SIZE, num_workers=PIPLINE_JOBS, test_img_flag=True):
    lab_wei_map_train = generate_image_patches(tr_img_folder, tr_img_lab_folder, save_folder, 'train', lab_fname_func, 
        targ_labs=targ_labs, stride=stride, patch_size=patch_size, num_workers=num_workers)
    lab_wei_map_valid = generate_image_patches(val_img_folder, val_img_lab_folder, save_folder, 'valid', lab_fname_func, 
        targ_labs=targ_labs, stride=stride, patch_size=patch_size, num_workers=num_workers) 
    if test_img_flag:
        lab_wei_map_test = generate_image_patches(te_img_folder, te_img_lab_folder, save_folder, 'test', lab_fname_func, 
            targ_labs=targ_labs, stride=stride, patch_size=patch_size, num_workers=num_workers) # We don't want to padding validation data sets
    else:
        lab_wei_map_test = None
    return lab_wei_map_train, lab_wei_map_valid, lab_wei_map_test


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
