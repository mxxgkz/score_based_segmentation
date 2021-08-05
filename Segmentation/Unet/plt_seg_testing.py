import os
import dill
import matplotlib.pyplot as plt
from datagenerator import *


def plt_seg_res(image, lab, pred, fig, num_samples, i, rand_line_colors, num_classes, plt_dir, title_size=20):
    img_size = image.shape[0]
    acc = np.sum(lab==pred)/img_size**2
    uni_labs = np.unique(lab)

    ax = fig.add_subplot(num_samples, 3, (i-1)*3+1)
    image = image - np.min(image)
    image = image/np.max(image)*255
    ax.imshow(image.astype(np.uint8), cmap='gray')
    ax.set_title("Texture labels in this plot: {}.".format(uni_labs), size=title_size)

    ax = fig.add_subplot(num_samples, 3, (i-1)*3+2)
    ax.imshow(lab, cmap='gray')
    ax.set_title("The true segmentation", size=title_size)

    ax = fig.add_subplot(num_samples, 3, (i-1)*3+3)
    ax.imshow(lab, cmap='gray')
    for j, l_idx in enumerate(list(range(num_classes))):
        coord_y, coord_x = np.where(pred==l_idx)
        ax.scatter(coord_x, coord_y, c=rand_line_colors[j%len(rand_line_colors)], marker='o', s=0.5, alpha=0.2)
    ax.set_title("The segementation with\nthe true: acc({:.4f})".format(acc), size=title_size)

# %% Validate on testing datasets
base_dir = os.path.expanduser('~/scratch/Data/texture/Brodatz/5_texture_images_5c/')
te_file = os.path.join(base_dir, 'test.txt')
batch_size = 32
num_classes = 5
te_data = UnetDataGenerator(te_file,
                            mode='inference',
                            batch_size=batch_size,
                            num_classes=num_classes)

log_dir = os.path.expanduser("~/scratch/logdir/Unet/")
checkpoint_path = os.path.join(log_dir, "finetune_unet/checkpoints")
model_name = 'rand_init_full_train_model_epoch120200130-221623.h5'

loaded_model = dill.load(open(os.path.join(checkpoint_path, model_name), 'rb'))

num_samples = 10

fig = plt.figure(figsize=(15, 6*num_samples), facecolor='w')

line_colors = ['blue', 'red', 'green', 'cyan', 'orange', 'magenta']

rand_line_colors = line_colors[:num_classes]
np.random.shuffle(rand_line_colors)

plt_dir = os.path.expanduser('Segmentation/Unet/')

for i, (image, lab) in enumerate(te_data.data.unbatch().take(num_samples),1):
    image, lab = image.numpy(), lab.numpy()

    pred = loaded_model(image).numpy()[0].argmax(axis=-1).astype(np.int32)

    plt_seg_res(image, lab, pred, fig, num_samples, i, rand_line_colors, num_classes, plt_dir)

plt.savefig(os.path.join(plt_dir, 'unet_seg_res_testing.png'))
plt.close()