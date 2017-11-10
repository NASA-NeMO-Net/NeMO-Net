from __future__ import unicode_literals
import json
import pytest
import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from NeMO_generator import NeMOImageGenerator, ImageSetLoader
# from voc2011.utils import increment_var

import yaml
with open("init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


@pytest.fixture
def is_same_shape():
    def f(shape_a, shape_b):
        for dim in shape_a:
            if dim and dim not in shape_b:
                return False
        return True
    return f


@pytest.fixture
def NeMO_datagen():
    datagen = NeMOImageGenerator(image_shape=[100, 100, 3],
                                 image_resample=True,
                                 pixelwise_center=True,
                                 pixel_mean=[104.00699, 116.66877, 122.67892])
    return datagen


@pytest.fixture
def NeMOtrain_loader():
    return ImageSetLoader(**init_args['image_set_loader']['train'])


@pytest.fixture
def val_loader():
    print(init_args['image_set_loader']['val'])
    return ImageSetLoader(**init_args['image_set_loader']['val'])

def test_flow_from_imageset(NeMO_datagen, NeMOtrain_loader, is_same_shape):
    train_generator = NeMO_datagen.flow_from_imageset(class_mode='categorical',
                                                     classes=4,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     image_set_loader=NeMOtrain_loader)
    for _ in range(5):
        batch_x, batch_y = train_generator.next(labelkey=[0,63,127,191])
        assert is_same_shape((1, 100, 100, 3), batch_x.shape)
        assert is_same_shape((1, 100, 100, 4), batch_y.shape)
        assert not np.all(batch_y == 0.)


#def test_loader(voc_datagen, voc_loader):
#    for fn in voc_loader.filenames:
#        x = voc_loader.load_img(fn)
#        x = voc_datagen.standardize(x)
#        y_true = voc_loader.load_seg(fn)
#        voc_loader.save(x, y_true, fn)
