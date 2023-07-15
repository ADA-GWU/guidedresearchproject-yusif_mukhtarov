import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pandas as pd
from scipy import ndimage as nd
from skimage.filters import sobel
import random

def feature_extraction(img):
    img = img.permute(1, 2, 0).numpy() 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img2 = img.reshape(-1)
    df = pd.DataFrame()
    df['Original Image'] = img2


    entropy_img = entropy(img, disk(1))

    entropy1 = entropy_img.reshape(-1)
    df['Entropy'] = entropy1

    gaussian_img = nd.gaussian_filter(img, sigma=3)

    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1


    
    sobel_img = sobel(img)

    sobel1 = sobel_img.reshape(-1)
    df['Sobel'] = sobel1
    return df, sobel_img


def sample_images(data_loader, number_of_samples_per_class):
    class_images = {}

    for batch in data_loader:
        images, labels = batch

        for image, label in zip(images, labels):
            if label.item() not in class_images:
                class_images[label.item()] = []
            class_images[label.item()].append(image)
    for class_label in class_images:
        if len(class_images[class_label]) < 10:
            raise ValueError(f"Class {class_label} does not have enough images.")

    sampled_images = {}
    for class_label in class_images:
        sampled_images[class_label] = random.sample(class_images[class_label], number_of_samples_per_class)

    return sampled_images



