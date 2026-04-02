import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class PetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset.
    Provides: image, class label (0-36), bbox [xc,yc,w,h] in pixels, segmentation mask.
    Images are normalized as required (mean/std ImageNet).
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, root: str, split: str = "train", img_size: int = 224,
                 transform=None, task: str = "all"):
        """
        root: path to oxford-iiit-pet directory
        split: 'train' or 'val' or 'test'
        task: 'classify' | 'localize' | 'segment' | 'all'
        """
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.task = task

        # Parse annotation file
        ann_file = os.path.join(root, "annotations", "list.txt")
        self.samples = []   # (image_name, class_id)
        with open(ann_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                name, class_id, species, breed_id = parts[0], int(
                    parts[1])-1, parts[2], parts[3]
                self.samples.append((name, class_id))

        # Split: 80% train, 10% val, 10% test (fixed seed)
        import random
        rng = random.Random(42)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        n = len(indices)
        if split == "train":
            self.indices = indices[:int(0.8*n)]
        elif split == "val":
            self.indices = indices[int(0.8*n):int(0.9*n)]
        else:
            self.indices = indices[int(0.9*n):]

        # Load bbox annotations
        self.bboxes = {}
        bbox_file = os.path.join(root, "annotations", "list.txt")
        # Actual head bounding boxes are in xmls folder
        self.xml_dir = os.path.join(root, "annotations", "xmls")

    def _load_bbox(self, name: str):
        """Load bbox from XML, return [xc, yc, w, h] in pixel space."""
        import xml.etree.ElementTree as ET
        xml_path = os.path.join(self.xml_dir, name + ".xml")
        if not os.path.exists(xml_path):
            return None
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            return None
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        # Scale to img_size
        size = root.find("size")
        orig_w = float(size.find("width").text)
        orig_h = float(size.find("height").text)
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        xmin *= scale_x
        xmax *= scale_x
        ymin *= scale_y
        ymax *= scale_y
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        return [xc, yc, w, h]

    def _preprocess_img(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.img_size, self.img_size))
        x = torch.from_numpy(np.array(img)).float() / 255.0
        x = x.permute(2, 0, 1)  # HWC → CHW
        mean = torch.tensor(self.MEAN).view(3, 1, 1)
        std = torch.tensor(self.STD).view(3, 1, 1)
        return (x - mean) / std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        name, class_id = self.samples[real_idx]

        img_path = os.path.join(self.root, "images", name + ".jpg")
        img = Image.open(img_path)
        img_tensor = self._preprocess_img(img)

        sample = {"image": img_tensor, "label": torch.tensor(
            class_id, dtype=torch.long)}

        if self.task in ("localize", "all"):
            bbox = self._load_bbox(name)
            if bbox is not None:
                sample["bbox"] = torch.tensor(bbox, dtype=torch.float32)
            else:
                sample["bbox"] = torch.zeros(4, dtype=torch.float32)

        if self.task in ("segment", "all"):
            mask_path = os.path.join(
                self.root, "annotations", "trimaps", name + ".png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).resize(
                    (self.img_size, self.img_size), Image.NEAREST)
                mask_arr = np.array(mask).astype(np.int64)
                # Trimap values: 1=foreground, 2=background, 3=border → remap to 0,1,2
                mask_arr = np.clip(mask_arr - 1, 0, 2)
                sample["mask"] = torch.tensor(mask_arr, dtype=torch.long)
            else:
                sample["mask"] = torch.zeros(
                    self.img_size, self.img_size, dtype=torch.long)

        return sample
