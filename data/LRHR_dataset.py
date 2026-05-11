from io import BytesIO
import lmdb
from PIL import Image
from osgeo import gdal
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np

def normalization(x):
    average = np.max(x) - np.min(x)
    return (x-np.min(x))/average

def read_image(image_path, maxdata):
    img = gdal.Open(image_path)
    b = np.array(img.ReadAsArray())
    return b/maxdata

def bilinear_interpolation(img, out_dim):
    channel, src_h, src_w = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 如果输入大小与原图大小相同，则返回原图
    dst_img = np.empty([channel, dst_h, dst_w], dtype=float)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 使用几何中心对称
                # 如果使用直接方式，src_x=dst_x*scale_x
                # scale是比例，通过同比例缩小/放大实现中心对齐
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找到将用于计算插值的点的坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算插值
                temp0 = (src_x1 - src_x) * img[i, src_y0, src_x0] + (src_x - src_x0) * img[i, src_y0, src_x1]
                temp1 = (src_x1 - src_x) * img[i, src_y1, src_x0] + (src_x - src_x0) * img[i, src_y1, src_x1]
                dst_img[i, dst_y, dst_x] = (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1

    return dst_img

class LRHRDataset(Dataset):
    def __init__(self, lr_path, hr_path, ref0_path, refsr0_path, ref1_path, refsr1_path, h=136, l=160, split='train', maxdata=255):
        self.h = h
        self.l = l
        self.split = split
        self.lr_path = Util.get_paths_from_images(lr_path)
        self.hr_path = Util.get_paths_from_images(hr_path)
        self.ref0_path = Util.get_paths_from_images(ref0_path)
        self.refsr0_path = Util.get_paths_from_images(refsr0_path)
        self.ref1_path = Util.get_paths_from_images(ref1_path)
        self.refsr1_path = Util.get_paths_from_images(refsr1_path)
        self.maxdata = maxdata

        self.data_len = len(self.hr_path)


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = read_image(self.hr_path[index],self.maxdata)/1.0
        img_HR = img_HR.astype(np.float32)
        img_LR = read_image(self.lr_path[index],self.maxdata)/1.0
        img_LR = img_LR.astype(np.float32)
        img_Ref0 = read_image(self.ref0_path[index],self.maxdata) / 1.0
        img_Ref0 = img_Ref0.astype(np.float32)
        img_RefSR0 = read_image(self.refsr0_path[index],self.maxdata) / 1.0
        img_RefSR0 = img_RefSR0.astype(np.float32)
        img_Ref1 = read_image(self.ref1_path[index],self.maxdata) / 1.0
        img_Ref1 = img_Ref1.astype(np.float32)
        img_RefSR1 = read_image(self.refsr1_path[index],self.maxdata) / 1.0
        img_RefSR1 = img_RefSR1.astype(np.float32)
        [img_LR, img_HR, img_Ref0, img_RefSR0, img_Ref1, img_RefSR1] = Util.transform_augment(
                [img_LR, img_HR, img_Ref0, img_RefSR0, img_Ref1, img_RefSR1], split=self.split, min_max=(-1, 1))
        return {'HR': img_HR, 'LR': img_LR, 'Ref0': img_Ref0, 'RefSR0': img_RefSR0, 'Ref1': img_Ref1, 'RefSR1': img_RefSR1, 'Index': index}

