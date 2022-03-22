import os
import numpy as np
import cv2
from PIL import Image
if __name__ == '__main__':
    imgs_dir = 'examples/content'
    fnames = set(os.listdir(imgs_dir))

    for fname in fnames:
        img_path = os.path.join(imgs_dir, fname)
        img_path_out = os.path.join('examples/content_blur_imgs', fname)

        # read the image with OpenCV
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = img + np.random.normal(loc=0.0, scale=0.08, size=img.shape)
        img *= 255.0
        img = np.clip(img, 0.0, 255.0)
        img = img.astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.save(img_path_out)
        #img.show()