#!/usr/bin/env python
# from https://github.com/aleju/imgaug

import imgaug as ia
from imgaug import augmenters as iaa


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


soft_seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        # horizontally flip 20% of all images
        # iaa.Fliplr(0.2),
        # # vertically flip 20% of all images
        # iaa.Flipud(0.2),
        # crop images by -5% to 5% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.05),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        iaa.Affine(
            # scale images to 100-200% of their size, individually per axis
            scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
            # translate by -20 to +20 percent (per axis)
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate by -30 to +30 degrees
            rotate=(-30, 30),
            # shear by -8 to +8 degrees
            shear=(-8, 8),
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            # if mode is constant, use a cval between 0 and 255
            cval=255
        ),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
                   [
                       # convert images into their superpixel representation
                       sometimes(
                           iaa.Superpixels(
                               p_replace=(0, 1.0),
                               n_segments=(10, 20)
                           )
                       ),
                       iaa.OneOf([
                           # blur images with a sigma between 0 and 1.0
                           iaa.GaussianBlur((0, 1.0)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.AverageBlur(k=(2, 7)),
                           # blur image using local medians with kernel sizes between 3 and 7
                           iaa.MedianBlur(k=(3, 7)),
                       ]),
                       # sharpen images
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                       # emboss images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0.75, 1.5)),
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.1, 0.5)),
                           iaa.DirectedEdgeDetect(alpha=(0.1, 0.5), direction=(0.0, 0.5)),
                       ])),
                       # add gaussian noise to images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=False),
                       iaa.OneOf([
                           # randomly remove up to 10% of the pixels
                           iaa.Dropout((0.01, 0.1), per_channel=False),
                           # iaa.CoarseDropout((0.03, 0.2), size_percent=(0.02, 0.1), per_channel=0.2),
                       ]),
                       # invert color channels
                       iaa.Invert(0.1, per_channel=False),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Add((-10, 10), per_channel=False),
                       # change hue and saturation
                       iaa.AddToHueAndSaturation((-2, 2)),
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           iaa.Multiply((0.5, 1.5)),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               first=iaa.Multiply((0.5, 1.5)),
                               second=iaa.ContrastNormalization((0.5, 2.0))
                           )
                       ]),
                       # improve or worsen the contrast
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=False),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.05)),
                       # sometimes move parts of the image around
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.1))),
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)


heavy_seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        # horizontally flip 50% of all images
        iaa.Fliplr(0.5),
        # vertically flip 20% of all images
        iaa.Flipud(0.2),
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate by -20 to +20 percent (per axis)
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate by -45 to +45 degrees
            rotate=(-45, 45),
            # shear by -16 to +16 degrees
            shear=(-16, 16),
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            # if mode is constant, use a cval between 0 and 255
            cval=(0, 255),
            # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            mode=ia.ALL
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
                   [
                       # convert images into their superpixel representation
                       sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                       iaa.OneOf([
                           # blur images with a sigma between 0 and 3.0
                           iaa.GaussianBlur((0, 3.0)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.AverageBlur(k=(2, 7)),
                           # blur image using local medians with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                       ]),
                       # sharpen images
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                       # emboss images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       ])),
                       # add gaussian noise to images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       iaa.OneOf([
                           # randomly remove up to 10% of the pixels
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       # invert color channels
                       iaa.Invert(0.05, per_channel=True),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change hue and saturation
                       iaa.AddToHueAndSaturation((-20, 20)),
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               second=iaa.ContrastNormalization((0.5, 2.0))
                           )
                       ]),
                       # improve or worsen the contrast
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       # sometimes move parts of the image around
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)