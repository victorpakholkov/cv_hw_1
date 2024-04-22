# https://docs.voxel51.com/user_guide/app.html
import os
import fiftyone as fo
import fiftyone.zoo as foz
import torch
from PIL import Image

""" Full split stats

    Train split: 118,287 images

    Test split: 40,670 images

    Validation split: 5,000 images """

# presuming that cwd is CV_HW_1
cv_dir = os.getcwd()
data_dir = os.path.join(cv_dir, "data")
fo.config.dataset_zoo_dir = data_dir

# выберите sample сами, если, весь датасет -144.1 GB + 1.9 GB, грузить его так, весь - пиздец, но если у кого-то есть возможность - было бы здорово # noqa

def load_coco(max_samples):
    dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits = ["train", "test", "validation"],
    label_types = ["detections"],
    # classes = classes
    max_samples = max_samples,)
    
    dataset.compute_metadata()
    return dataset

