"""Transparent Semantic Segmentation Dataset."""
import os
import logging
import torch
import numpy as np

from PIL import Image
from .seg_data_base import SegmentationDataset


class TransparentSegmentationSAM(SegmentationDataset):
    BASE_DIR = 'Trans10K_cls12'
    NUM_CLASS = 12

    def __init__(self, root='datasets/Grounded_ID2', split='test', mode=None, transform=None, **kwargs):
        super(TransparentSegmentationSAM, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(self.root, self.BASE_DIR)
        assert os.path.exists(root), "Please put the data in {SEG_ROOT}/datasets/transparent"
        self.images, self.masks = _get_trans10k_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        logging.info('Found {} images in the folder {}'.format(len(self.images), root))

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _val_sync_transform_resize(self, img, mask):
        short_size = self.crop_size
        img = img.resize(short_size, Image.BILINEAR)
        mask = mask.resize(short_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index]).convert("P")
        # mask = Image.open(self.masks[index]).convert("L")
        # for i in range(0, mask.size[0]):
        #     for j in range(0, mask.size[1]):
        #         if mask.getpixel((i, j)) == 255:
        #             mask.putpixel(((i, j)), 8)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ('Background', 'Shelf', 'Jar or Tank', 'Freezer', 'Window',
                'Glass Door', 'Eyeglass', 'Cup', 'Floor Glass', 'Glass Bow',
                'Water Bottle', 'Storage Box')


def _get_trans10k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'train/images')
        mask_folder = os.path.join(folder, 'train/masks_12')
    elif mode == "val":
        img_folder = os.path.join(folder, 'validation/images')
        mask_folder = os.path.join(folder, 'validation/masks_12')
    else:
        assert  mode == "test"
        img_folder = os.path.join(folder, 'test/images')
        mask_folder = os.path.join(folder, 'test/masks_12')

    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '_mask.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask:', maskpath)

    return img_paths, mask_paths


if __name__ == '__main__':
    train_dataset = TransparentSegmentation()
