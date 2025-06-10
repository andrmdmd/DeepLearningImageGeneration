import torchvision.transforms as T
import torch
import numpy as np
import random
import math


def get_ada_aug(p: float):
    """
    Returns a custom augmentation pipeline for adaptive augmentation based on the provided pseudocode.
    Args:
        p (float): Probability of applying the augmentation.
    Returns:
        torchvision.transforms.Compose: A composition of augmentations.
    """
    def random_x_flip(image):
        if random.random() < p:
            return T.functional.hflip(image)
        return image

    def random_90_rotation(image):
        if random.random() < p:
            rotations = [0, 90, 180, 270]
            angle = random.choice(rotations)
            return T.functional.rotate(image, angle)
        return image

    def random_translation(image):
        if random.random() < p:
            w, h = T.functional.get_image_size(image)
            tx = random.uniform(-0.125, 0.125) * w
            ty = random.uniform(-0.125, 0.125) * h
            return T.functional.affine(image, angle=0, translate=(round(tx), round(ty)), scale=1, shear=0)
        return image

    def random_isotropic_scaling(image):
        if random.random() < p:
            s = random.lognormvariate(0, (0.2 * math.log(2))**2)
            return T.functional.affine(image, angle=0, translate=(0, 0), scale=s, shear=0)
        return image

    def random_pre_rotation(image):
        prot = 1 - math.sqrt(1 - p)
        if random.random() < prot:
            theta = random.uniform(-math.pi, math.pi)
            return T.functional.rotate(image, math.degrees(-theta))
        return image

    def random_post_rotation(image):
        prot = 1 - math.sqrt(1 - p)
        if random.random() < prot:
            theta = random.uniform(-math.pi, math.pi)
            return T.functional.rotate(image, math.degrees(-theta))
        return image

    def random_brightness(image):
        if random.random() < p:
            b = random.gauss(0, 0.2)
            return T.functional.adjust_brightness(image, 1 + b)
        return image

    def random_contrast(image):
        if random.random() < p:
            c = random.lognormvariate(0, 0.5 * math.log(2))
            return T.functional.adjust_contrast(image, c)
        return image

    def random_hue(image):
        if random.random() < p:
            theta = random.uniform(-math.pi, math.pi)
            return T.functional.adjust_hue(image, theta / (2*math.pi))
        return image

    def random_saturation(image):
        if random.random() < p:
            s = random.lognormvariate(0, math.log(2))
            return T.functional.adjust_saturation(image, s)
        return image

    return T.Compose([
        T.Lambda(random_x_flip),
        T.Lambda(random_90_rotation),
        T.Lambda(random_translation),
        T.Lambda(random_isotropic_scaling),
        T.Lambda(random_pre_rotation),
        T.Lambda(random_post_rotation),
        T.Lambda(random_brightness),
        T.Lambda(random_contrast),
        T.Lambda(random_hue),
        T.Lambda(random_saturation),
    ])
