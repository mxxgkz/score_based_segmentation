# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# %%

def save_png(base_dir, res_path):
    img = Image.open(os.path.join(base_dir, res_path))
    img.save(os.path.join(base_dir, res_path[:-4]+'.png'))
    img_png = Image.open(os.path.join(base_dir, res_path[:-4]+'.png'))
    plt.imshow(img_png, cmap='gray')


# %%
base_dir = '/projects/p30309/Data/texture/Prof_Brinson/TEM_images/'

# %%
res_path = '100_RPM/100_KJ_KG/019.tif' 
save_png(base_dir, res_path)

# %%
res_path = '300_RPM/4000_KJ_KG/005.tif'
save_png(base_dir, res_path)

# %%
res_path = '100_RPM/100_KJ_KG/017.tif' 
save_png(base_dir, res_path)

# %%
res_path = '100_RPM/100_KJ_KG/015.tif' 
save_png(base_dir, res_path)

# %%
