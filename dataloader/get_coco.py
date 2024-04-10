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

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train", "test", "validation"],
    label_types=["detections"],
    max_samples=5000,
)

classes = dataset.distinct("ground_truth.detections.label")
id2name, name2id = {}, {}
classes_id_list = []
for class_id, class_name in enumerate(classes):
    # if class_name in classes_list:
    # classes_id_list.append(class_id)
    id2name[class_id] = class_name
    name2id[class_name] = class_id


class from51totorch(torch.utils.data.Dataset):

    def __init__(
        self,
        fiftyone_dataset,
        S=7,
        B=2,
        C=5,
        transform=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.S = S
        self.B = B
        self.C = C

        self.samples = fiftyone_dataset
        self.transform = transform
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )  # noqa

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fo.utils.coco.COCOObject.from_label(
                det,
                metadata,
                category_id=category_id,
            )
            if coco_obj.category_id not in classes_id_list:
                continue
            # class_label = coco_obj.category_id - 1
            class_label = classes_id_list.index(coco_obj.category_id)
            x, y, w, h = coco_obj.bbox
            x_center = (x + w / 2) / img.size[0]
            y_center = (y + h / 2) / img.size[1]
            width = w / img.size[0]
            height = h / img.size[1]
            boxes.append([class_label, x_center, y_center, width, height])

        boxes = torch.tensor(boxes)

        if self.transform:
            img, boxes = self.transform(img, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            cell_x = self.S * x - j
            cell_y = self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor(
                    [cell_x, cell_y, width_cell, height_cell]
                )
                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return img, label_matrix

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes
