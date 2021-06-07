# %% [markdown]
# Extract links from a page
import mechanize
from time import sleep
import chardet
import os
from urllib.parse import urljoin

# Download folders
dnld_dir = os.path.expanduser('~/scratch/Data/texture/Kylberg/')
if not os.path.exists(dnld_dir):
    os.mkdir(dnld_dir)

#Make a Browser (think of this as chrome or firefox etc)
br = mechanize.Browser()

#visit http://stockrt.github.com/p/emulating-a-browser-in-python-with-mechanize/
#for more ways to set up your br browser object e.g. so it look like mozilla
#and if you need to fill out forms with passwords.

# Open your site
br.open('http://www.cb.uu.se/~gustaf/texture/data/with-rotations-zip/')

f=open(os.path.join(dnld_dir, "source.txt"),"w")
resp = br.response().read()
print("The encoding of the response is {}.".format(chardet.detect(resp)['encoding']))
f.write(resp.decode("utf-8")) #can be helpful for debugging maybe
f.close()

filetypes=[".zip"] #you will need to do some kind of pattern matching on your files
myfiles=[]
for l in br.links(): #you can also iterate through br.forms() to print forms on the page!
    for t in filetypes:
        if t in str(l): #check if this link has the file extension we want (you may choose to use reg expressions or something)
            myfiles.append(l)
print(myfiles)

# Write file names to a txt file
fnames = []
f_fnames=open(os.path.join(dnld_dir, "fnames.txt"), "w")
for l in myfiles:
    f_fnames.write(l.text+"\n")
    fnames.append(l.text)
f_fnames.close()


# %% [markdown]
def downloadlink(l):
    f=open(os.path.join(l.text[:-4]+'.txt'),"w") #perhaps you should open in a better way & ensure that file doesn't already exist.
    # br.click_link(l)
    # br.follow_link(urljoin(l.base_url, l.text))
    f.write(os.popen('wget {} -O {}'.format(urljoin(l.base_url, l.text), os.path.join(dnld_dir, l.text))).read())
    print(l.text," has been downloaded")
    f.close()
    #br.back()

for l in myfiles:
    sleep(10) #throttle so you dont hammer the site
    downloadlink(l)
    # break

# %%
for fn in fnames:
    os.popen('mkdir {}'.format(os.path.join(dnld_dir, fn[:-4])))
    os.popen('unzip {} -d {}'.format(os.path.join(dnld_dir, fn), os.path.join(dnld_dir, fn[:-4]))).read()

# %%
for fn in fnames:
    os.popen('rm {}'.format(os.path.join(dnld_dir, fn)))


# %% [markdown]
## Plot all samples with labels together
import matplotlib.pyplot as plt
import cv2

fig = plt.figure(figsize=(30, 18))
for i, fn in enumerate(fnames):
    fig.add_subplot(4,7,i+1)
    fn_dir = os.path.join(dnld_dir, fn[:-4])
    for ff in os.listdir(fn_dir):
        one_file = ff
        break
    img = cv2.imread(os.path.join(fn_dir, one_file))
    plt.imshow(img)
    plt.title(fn[:-4])
    plt.axis('off')
plt.savefig(os.path.join(dnld_dir, 'example_imgs_labs.png'))
plt.show()

# %%

