import cv2
import random
import numpy as np

__all__ = [
    "Compose",
    "Normalize",
    "CenterCrop",
    "RgbToGray",
    "RandomCrop",
    "HorizontalFlip",
    "AddNoise",
    "Resize",
    "NormalizeUtterance",
]


class Compose(object):
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RgbToGray(object):
    def __call__(self, frames):
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.0)
        delta_h = int(round((h - th)) / 2.0)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class HorizontalFlip(object):
    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


class Resize(object):
    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        return cv2.resize(
            frames,
            dsize=(self.flip_ratio, self.flip_ratio),
            interpolation=cv2.INTER_CUBIC,
        )
