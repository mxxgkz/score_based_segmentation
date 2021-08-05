
# coding: utf-8

# In[1]:

from unet import *
import tensorflow as tf


# ### Fix bugs
# One quest, I found difficulty in using `plot_model`. I then follow this [link](https://datascience.stackexchange.com/a/37431/89898). The main point is that we need to link the library of `graphviz`. I clean the path in `~/.bashrc`. And I use `conda install python-graphviz`, install of just `pip install --user graphviz`. Then it works.

# In[3]:

temp_model = Unet(0.5, 5, [], plot_model=True)
tf.keras.utils.plot_model(temp_model.model.graph, 'unet_plot.png', show_shapes=True, expand_nested=True)


# In[4]:

temp_model.model.graph.summary()


# In[ ]:



