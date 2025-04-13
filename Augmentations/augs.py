import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if np.random.rand() < self.prob:
            return np.flip(image, axis=-1)  # Flip along width axis
        return image

class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if np.random.rand() < self.prob:
            return np.flip(image, axis=-2)  # Flip along height axis
        return image

class RandomRotation:
    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, image):
        angle = np.random.choice(self.angles)
        if angle == 0:
            return image
        return np.rot90(image, k=angle // 90, axes=(-2, -1))

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        noise = np.random.normal(self.mean, self.std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0.0, 1.0)

class Normalize:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

class ToFloat:
    def __call__(self, image):
        return image.astype(np.float32) / 255.0
