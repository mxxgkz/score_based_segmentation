# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, parallel_backend, delayed
# np.random.seed(123)




# %%
base_dir = os.path.expanduser('/projects/p30309/neurips2021/Data/texture/Kylberg_images/')
class_names = []
for fn in os.listdir(base_dir):
    if fn.endswith('rotations'):
        class_names.append(fn.split('-')[0])
print(class_names)


# %%
# # save class names and labels
# cn_lab = {}
# f=open(os.path.join(base_dir, 'class_names.txt'), 'w')
# for i, cn in enumerate(class_names):
#     f.write(cn+' '+str(i)+'\n')
#     cn_lab[cn] = i
# f.close()

cn_lab = {}
lab_cn = {}
lab_cn_path = os.path.join(base_dir, 'class_names.txt')
with open(lab_cn_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.strip().split(' ')
        cn_lab[items[0]] = items[1]
        lab_cn[items[1]] = items[0]
print(cn_lab)

# %%
num_classes = 28
rotation = np.random.randint(low=0, high=12, size=(num_classes,))
print(rotation)

# %%
# http://www.cb.uu.se/~gustaf/texture/
# Those patches of image are not overlapped.
# The rotation of those patches may create some overlapping between patches, but not completely overlapped.
# Each class has four big images. Each images can generate 40 non-overlapped patches. Each patch can have 12 
# rotations. In total, we have 53760 images with size 576*576

# sample_lab = ['a','b','c','d']
# ls_data_rows = []
# for ci, cn in enumerate(class_names):
#     for lab in sample_lab:
#         for pidx in range(1,41):
#             pidx_str = str(pidx)
#             pidx_str = 'p'+'0'*(3-len(pidx_str))+pidx_str
#             rot_str = str(int(30*rotation[ci]))
#             rot_str = 'r'+'0'*(3-len(rot_str))+rot_str
#             cls_dir = os.path.join(base_dir, cn+'-with-rotations')
#             ls_data_rows.extend([[os.path.join(cls_dir, '-'.join([cn, lab, pidx_str, rot_str])+'.png'), str(cn_lab[cn]), str(j)] for j in range(4)]) # Break each patch further to 4 smaller non-overlapped patches

# print(len(ls_data_rows))

sample_lab = ['a','b','c','d']

# for ci, cn in enumerate(class_names):
#     for lab in sample_lab:
#         for pidx in range(1,41):
#             pidx_str = str(pidx)
#             pidx_str = 'p'+'0'*(3-len(pidx_str))+pidx_str
#             rot_str = str(int(30*rotation[ci]))
#             rot_str = 'r'+'0'*(3-len(rot_str))+rot_str
#             cls_dir = os.path.join(base_dir, cn+'-with-rotations')
#             ls_row = [[os.path.join(cls_dir, '-'.join([cn, lab, pidx_str, rot_str])+'.png'), str(cn_lab[cn]), str(j)] for j in range(4)] # Break each patch further to 4 smaller non-overlapped patches
#             rnd_num = np.random.randint(10)
#             if rnd_num<8:
#                 ls_train_data_rows.extend(ls_row) 
#             elif rnd_num==8:
#                 ls_valid_data_rows.extend(ls_row)
#             else:
#                 ls_test_data_rows.extend(ls_row)

def proc_one_class(base_dir, ci, cn, sample_lab, num_patch, num_rot, cn_lab):
    num_sample = num_patch*len(sample_lab)
    ls_set_flag = [0]*int(num_sample*0.8) + [1]*int(num_sample*0.1) + [2]*int(num_sample*0.1)
    ls_set_flag += [2]*(num_sample-len(ls_set_flag))
    arr_set_flag = np.array(ls_set_flag)
    ls_train_data_rows, ls_valid_data_rows, ls_test_data_rows = [], [], []
    for lab in sample_lab:
        np.random.shuffle(arr_set_flag) # if one patch in the test data set, all its rotations are in test data set
        for pidx in range(1,num_patch+1):
            for rot in range(num_rot):
                pidx_str = str(pidx)
                pidx_str = 'p'+'0'*(3-len(pidx_str))+pidx_str
                rot_str = str(int(30*rot))
                rot_str = 'r'+'0'*(3-len(rot_str))+rot_str
                cls_dir = os.path.join(base_dir, cn+'-with-rotations')
                ls_row = [[os.path.join(cls_dir, '-'.join([cn, lab, pidx_str, rot_str])+'.png'), str(cn_lab[cn]), str(j)] for j in range(4)] # Break each patch further to 4 smaller non-overlapped patches
                if arr_set_flag[pidx-1]==0:
                    ls_train_data_rows.extend(ls_row) 
                elif arr_set_flag[pidx-1]==1:
                    ls_valid_data_rows.extend(ls_row)
                else:
                    ls_test_data_rows.extend(ls_row)
    return ls_train_data_rows, ls_valid_data_rows, ls_test_data_rows

def write_txt(fname, fdir, ls_rows, ls_idx=None):
    if ls_idx is None:
        ls_idx = list(range(len(ls_rows)))
        np.random.shuffle(ls_idx)
    f=open(os.path.join(fdir, fname),'w')
    for idx in ls_idx:
        row = ls_rows[idx]
        f.write(','.join(row)+'\n')
    f.close()

def flatten_ls(ls_ls, shuffle=True):
    ls_flatten = []
    for ls in ls_ls:
        ls_flatten.extend(ls)
    if shuffle:
        np.random.shuffle(ls_flatten)
    return ls_flatten

# %%
num_patch = 40
num_rot = 12
ls_tasks = [(base_dir, ci, cn, sample_lab, num_patch, num_rot, cn_lab) for ci, cn in enumerate(class_names)]
with parallel_backend('loky', n_jobs=8):
    res = Parallel(verbose=10, pre_dispatch='2*n_jobs')(delayed(proc_one_class)(*task) for task in ls_tasks)

ls_ls_train_data_rows, ls_ls_valid_data_rows, ls_ls_test_data_rows = zip(*res)
ls_train_data_rows = flatten_ls(ls_ls_train_data_rows)
ls_valid_data_rows = flatten_ls(ls_ls_valid_data_rows)
ls_test_data_rows = flatten_ls(ls_ls_test_data_rows)


# %%
# row_idx = list(range(len(ls_data_rows)))
# np.random.shuffle(row_idx)
# train_ratio = 0.9
# bdidx = int(len(ls_data_rows)*0.9)
# ls_train_idx = row_idx[:bdidx]
# ls_valid_idx = row_idx[bdidx:]

# write_txt('train.txt', base_dir, ls_data_rows, ls_train_idx)
# write_txt('valid.txt', base_dir, ls_data_rows, ls_valid_idx)
write_txt('train.txt', base_dir, ls_train_data_rows)
write_txt('valid.txt', base_dir, ls_valid_data_rows)
write_txt('test.txt', base_dir, ls_test_data_rows)


# %%
fig = plt.figure(figsize=(42,26), facecolor='w')
class_names.sort()
for ci, cn in enumerate(class_names):
    img_arr = np.array(Image.open(os.path.join(base_dir, cn+'-with-rotations/'+'-'.join([cn,'a','p001','r090'])+'.png')))
    img_arr = (img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr))*255
    ax = fig.add_subplot(4, 7, ci+1)
    ax.imshow(img_arr.astype(np.uint8), cmap=plt.cm.gray)
    ax.set_title(cn, size=35)

plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'example_imgs_labs.png'))
plt.show()

# %%
## Prepare train files for google drive, smaller data set
def change_delimiter(fname, new_fname):
    with open(os.path.join(base_dir, new_fname), 'w') as new_f:
        with open(os.path.join(base_dir, fname), 'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            new_f.write(','.join(items))

change_delimiter('train.txt', 'train_gd.txt')
change_delimiter('valid.txt', 'valid_gd.txt')

# %% [markdown]
## Copy those files in train and valid to another folder
new_base_dir = '/home/ghhgkz/scratch/Data/texture/Kylberg-small/'
for cn in class_names:
    os.mkdir(os.path.join(new_base_dir, cn+'-with-rotations'))

# %% [markdown]
## Function of copy images.
def copy_images(fname):
    with open(os.path.join(base_dir, fname), 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            os.popen('cp {} {}'.format(items[0], items[0].replace('Kylberg', 'Kylberg-small')))

# %%
copy_images('train.txt')
copy_images('valid.txt')


# %% [markdown]
## Check if all folders has enough images
for i, folder in enumerate(os.listdir(new_base_dir)):
    print(i, os.popen('ls -1 {} | wc -l'.format(os.path.join(new_base_dir, folder))).read())

# %%
