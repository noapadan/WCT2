import torch
#import kornia
import argparse
import cv2
import numpy as np

import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt


'''
def denoise_image(noisy_image, verbose=False):
    image = cv2.imread('examples/content_blur/in14.png')
    dst = cv2.fastNlMeansDenoisingColored(np.array(image), None, 11, 6, 7, 21)
    cv2.imshow('img', dst)
    cv2.waitKey(0)

    tv_denoiser = TVDenoise(noisy_image)

    # define the optimizer to optimize the 1 parameter of tv_denoiser
    optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr=0.8, momentum=0.9)

    # run the optimization loop
    num_iters = 500
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = tv_denoiser()
        if i % 25 == 0:
            print("Loss in iteration {} of {}: {:.3f}".format(i, num_iters, loss.item()))
        loss.backward()
        optimizer.step()

    # convert back to numpy
    img_clean: np.ndarray = kornia.tensor_to_image(tv_denoiser.get_clean_image())

    # Create the plot
    if verbose:
        cv2.imshow('img',img_clean)
        cv2.waitKey(0)

    return tv_denoiser.get_clean_image()

# define the total variation denoising network
class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        self.regularization_term = kornia.losses.TotalVariation()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image

    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + 0.0001 * self.regularization_term(self.clean_image)

    def get_clean_image(self):
        return self.clean_image
'''
def deep_image_prior_denoising_tensor(blur_image_tensor):
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=3, pretrained=False)
    lr = 0.005
    iterations = 250
    check_cycle = 25  # How often to check image restoration
    blur_image_tensor = blur_image_tensor.detach()
    #z = torch.randn(3, 256, 256, requires_grad=False).unsqueeze(0)
    z = torch.randn(blur_image_tensor.shape, requires_grad=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        blur_image_tensor = blur_image_tensor.to('cuda')
        model = model.to('cuda')
        z = z.to('cuda')

    # for re running - initilzie new z
    z = torch.randn(blur_image_tensor.shape, requires_grad=False).to(device)
    #
    restored_images = []
    for iteration in range(iterations + 1):
        print(iteration)
        optimizer.zero_grad()
        x_star = model(z)
        loss = criterion(x_star, blur_image_tensor)#.requires_grad_(True)
        loss.backward()
        optimizer.step()

        if iteration % check_cycle == 0:
            restored_images.append(transforms.ToPILImage()(x_star[0].to('cpu')))
            print("restored after iteration: " + str(iteration))

            #open_cv_image = np.array(x_star[0].to('cpu').detach())
            # Convert RGB to BGR
            #open_cv_image = open_cv_image[:, :, ::-1].copy()
            #cv2.imshow('blur', open_cv_image)
            #cv2.waitKey(0)
            pl_img = transforms.ToPILImage()(x_star[0].to('cpu').detach())
            pl_img.save("./output_DIP/img"+str(iteration)+".png")
def deep_image_prior_denoising(blur_image):
    blur_image = Image.open(blur_image)
    #blur_image = cv2.resize(blur_image, (256,256))
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=3, pretrained=False)
    #f16_noisy = Image.open(f16_noisy)
    #cv2.imshow('blur img', blur_image)
    #cv2.waitKey(0)
    #noisy_image = torch.tensor(blur_image)#.resize((3,256,256))
    #noisy_image = torch.reshape(noisy_image, (3,256,256))
    #noisy_image_batch = noisy_image.unsqueeze(0)
    #blur_image = blur_image.resize((256,256))
    blur_image.save("./output_DIP/img_blur.png")
    noisy_image_tensor = transforms.ToTensor()(blur_image)
    noisy_image_batch = noisy_image_tensor.unsqueeze(0)

    lr = 0.005
    iterations = 250
    check_cycle = 25  # How often to check image restoration

    #z = torch.randn(3, 256, 256, requires_grad=False).unsqueeze(0)
    z = torch.randn(3, blur_image.size[1], blur_image.size[0], requires_grad=False).unsqueeze(0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        noisy_image_batch = noisy_image_batch.to('cuda')
        model = model.to('cuda')
        z = z.to('cuda')

    # for re running - initilzie new z
    z = torch.randn(3, blur_image.size[1], blur_image.size[0], requires_grad=False).unsqueeze(0).to(device)
    #
    restored_images = []
    for iteration in range(iterations + 1):
        print(iteration)
        optimizer.zero_grad()
        x_star = model(z)
        loss = criterion(x_star, noisy_image_batch)
        loss.backward()
        optimizer.step()

        if iteration % check_cycle == 0:
            restored_images.append(transforms.ToPILImage()(x_star[0].to('cpu')))
            print("restored after iteration: " + str(iteration))

            #open_cv_image = np.array(x_star[0].to('cpu').detach())
            # Convert RGB to BGR
            #open_cv_image = open_cv_image[:, :, ::-1].copy()
            #cv2.imshow('blur', open_cv_image)
            #cv2.waitKey(0)
            pl_img = transforms.ToPILImage()(x_star[0].to('cpu').detach())
            pl_img.save("./output_denoise/img"+str(iteration)+".png")
if __name__ == '__main__':
    #blur_image = cv2.imread('outputs_blur/in00_cat5_decoder..png')
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default=None)
    #parser.add_argument('--out_dir', type=str, default=None)
    config = parser.parse_args()

    print(config)
    deep_image_prior_denoising(blur_image=config.img)
    #deep_image_prior_denoising(blur_image='/Users/firas/noa/WCT2/outputs_style_blur/in00_cat5_decoder..png')

    #deep_image_prior_denoising(blur_image='/Users/firas/noa/WCT2/outputs_style_blur/in00_cat5_decoder..png')

    # read the image with OpenCV
    #img: np.ndarray = cv2.imread('examples/content_blur/in14.png')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    #img = img + np.random.normal(loc=0.0, scale=0.1, size=img.shape)
    #img = np.clip(img, 0.0, 1.0)

    # convert to torch tensor
    #noisy_image: torch.tensor = kornia.image_to_tensor(img).squeeze()  # CxHxW
    #denoise_image(noisy_image, verbose=True)
