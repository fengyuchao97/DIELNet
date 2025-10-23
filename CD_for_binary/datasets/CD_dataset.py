"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data

from datasets.data_utils import CDDataAugmentation

from scipy import signal

def fft(img):
    return np.fft.fft2(img)

def fftshift(img):
    return np.fft.fftshift(fft(img))

def ifft(img):
    return np.fft.ifft2(img)

def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask

def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:,i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result

def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))
    return np.array(Images_freq_low)

# def generateDataWithDifferentFrequencies_3Channel(Images, r):
#     Images_freq_low = []
#     Images_freq_high = []

#     print(Images.shape)
#     mask = mask_radial(np.zeros([Images.shape[2], Images.shape[3]]), r)
#     for i in range(Images.shape[0]):
#         tmp = np.zeros([3, Images.shape[2], Images.shape[3]])
#         for j in range(3):
#             fd = fftshift(Images[i, j, :, :])
#             fd = fd * mask
#             img_low = ifftshift(fd)
#             tmp[j,:,:] = np.real(img_low)
#         Images_freq_low.append(tmp)
#         tmp = np.zeros([Images.shape[2], Images.shape[3], 3])
#         for j in range(3):
#             fd = fftshift(Images[i, j, :, :])
#             fd = fd * (1 - mask)
#             img_high = ifftshift(fd)
#             tmp[j,:,:] = np.real(img_high)
#         Images_freq_high.append(tmp)

#     return np.array(Images_freq_low), np.array(Images_freq_high)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[0], Images.shape[1]]), r)
    tmp = np.zeros([Images.shape[0], Images.shape[1], 3])
    for j in range(3):
        fd = fftshift(Images[:, :, j])
        fd = fd * mask
        img_low = ifftshift(fd)
        tmp[:,:,j] = np.real(img_low)
    Images_freq_low.append(tmp)
    tmp = np.zeros([Images.shape[0], Images.shape[1], 3])
    for j in range(3):
        fd = fftshift(Images[:, :, j])
        fd = fd * (1 - mask)
        img_high = ifftshift(fd)
        tmp[:,:,j] = np.real(img_high)
    Images_freq_high.append(tmp)

    return np.array(Images_freq_low).astype(np.uint8), np.array(Images_freq_high).astype(np.uint8)

def swapFrequencies_3Channel(Images1, Images2, r):
    """交换两个输入图像的高频分量，生成新的图像"""
    mask = mask_radial(Images1[:, :, 0], r)
    
    # 处理第一个输入图像
    tmp_low1 = np.zeros_like(Images1)
    tmp_high1 = np.zeros_like(Images1)
    for j in range(3):
        fd1 = fftshift(fft(Images1[:, :, j]))
        fd1_low = fd1 * mask
        fd1_high = fd1 * (1 - mask)
        
        img_low1 = ifft(ifftshift(fd1_low))
        img_high1 = ifft(ifftshift(fd1_high))
        
        tmp_low1[:, :, j] = np.real(img_low1)
        tmp_high1[:, :, j] = np.real(img_high1)
    
    # 处理第二个输入图像
    tmp_low2 = np.zeros_like(Images2)
    tmp_high2 = np.zeros_like(Images2)
    for j in range(3):
        fd2 = fftshift(fft(Images2[:, :, j]))
        fd2_low = fd2 * mask
        fd2_high = fd2 * (1 - mask)
        
        img_low2 = ifft(ifftshift(fd2_low))
        img_high2 = ifft(ifftshift(fd2_high))
        
        tmp_low2[:, :, j] = np.real(img_low2)
        tmp_high2[:, :, j] = np.real(img_high2)
    
    # 交换高频分量
    tmp_X_new = np.zeros_like(Images1)
    tmp_Y_new = np.zeros_like(Images2)
    
    for j in range(3):
        fd1_low = fftshift(fft(tmp_low1[:, :, j]))
        fd2_high = fftshift(fft(tmp_high2[:, :, j]))
        
        merged_X = ifft(ifftshift(fd1_low + fd2_high))
        
        fd2_low = fftshift(fft(tmp_low2[:, :, j]))
        fd1_high = fftshift(fft(tmp_high1[:, :, j]))
        
        merged_Y = ifft(ifftshift(fd2_low + fd1_high))
        
        tmp_X_new[:, :, j] = np.real(merged_X)
        tmp_Y_new[:, :, j] = np.real(merged_Y)
    
    # return tmp_X_new.astype(np.uint8), tmp_Y_new.astype(np.uint8)#, tmp_low1, tmp_high1, tmp_low2, tmp_high2
    return np.transpose(tmp_X_new, (1, 2, 0)).astype(np.uint8), np.transpose(tmp_Y_new, (1, 2, 0)).astype(np.uint8)#, np.transpose(tmp_low1, (0, 1, 2)), np.transpose(tmp_high1, (0, 1, 2)), np.transpose(tmp_low2, (0, 1, 2)), np.transpose(tmp_high2, (0, 1, 2))


"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label_mul"
ANNOT_FOLDER_NAME_BINARY = "label"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))

def get_binary_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME_BINARY, img_name.replace('.jpg', label_suffix))
    
class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        img_low, img_high = generateDataWithDifferentFrequencies_3Channel(img, 4)
        img_B_low, img_B_high = generateDataWithDifferentFrequencies_3Channel(img_B, 4)

        # [img_ori, img_B_ori], _ = self.augm.to_tensor([img, img_B],[])
        # [img_weak, img_B_weak], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)
        [img_weak, img_B_weak], _ = self.augm.transform([img_low, img_B_low],[], to_tensor=self.to_tensor)

        [img_strong, img_B_strong] = self.augm.transform_strong([img_weak, img_B_weak],[])
        [img_strong2, img_B_strong2] = self.augm.transform_strong2([img_weak, img_B_weak],[])
        [img_strong3, img_B_strong3] = self.augm.transform_strong3([img_weak, img_B_weak],[])

        # return {'A': img, 'B': img_B, 'name': name}
        # img_aug = torch.stack([img_strong,img_weak,img_ori,img_strong2],dim=1)
        # img_B_aug = torch.stack([img_B_strong2,img_B_ori,img_B_weak,img_B_strong],dim=1)

        # img_aug = torch.stack([img_ori,img_weak,img_strong,img_strong2],dim=1)
        # # print('fyc_stack:',img.shape)
        # img_B_aug = torch.stack([img_B_ori,img_B_weak,img_B_strong,img_B_strong2],dim=1)

        # img_aug = torch.stack([img_ori,img_B_ori,img_weak,img_B_weak],dim=1)
        # img_B_aug = torch.stack([img_strong,img_B_strong,img_strong2,img_B_strong2],dim=1)
        
        img_aug = torch.stack([img_strong3, img_B_strong3, img_weak, img_B_weak, img_strong,img_B_strong,img_strong2,img_B_strong2],dim=1)

        return {'A': img_weak, 'B': img_B_weak, 'name': name, 'aug': img_aug}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size

# # 定义颜色映射字典
# color_to_class = {
#     (0, 0, 0): 0,  # 黑色对应类别 1
#     (1, 1, 0): 1,  # 黄色对应类别 2
#     (1, 0, 0): 2,  # 红色对应类别 3
#     (0, 0, 1): 3   # 蓝色对应类别 4
# }

# color_to_class2 = {
#     (0, 0, 0): 0,  # 黑色对应类别 1
#     (1, 1, 0): 2,  # 黄色对应类别 2
#     (1, 0, 0): 1,  # 红色对应类别 3
#     (0, 0, 1): 3   # 蓝色对应类别 4
# }

# 定义颜色映射字典
color_to_class = {
    (0, 0, 0): 0,  # 黑色对应类别 1
    (1, 1, 0): 1,  # 黄色对应类别 2
    (1, 0, 0): 2  # 红色对应类别 3
}


import random
def random_unit(p: float):
    if p == 0:
        return False
    if p == 1:
        return True

    R = random.random()
    if R < p:
        return True
    else:
        return False

class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True, data_name=None):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        L_path_binary = get_binary_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        
        label_binary = np.array(Image.open(L_path_binary), dtype=np.uint8)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label_binary = label_binary // 255

        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        label = np.array(Image.open(L_path), dtype=np.uint8)
        compute_semantic = True
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255

        r_values = [4, 8, 16, 32]

        [img_weak, img_B_weak], [label, label_binary] = self.augm.transform([img, img_B],[label, label_binary], to_tensor=self.to_tensor)
        
        img_strong, img_B_strong = swapFrequencies_3Channel(img_weak, img_B_weak, r_values[0])
        img_strong2, img_B_strong2 = swapFrequencies_3Channel(img_weak, img_B_weak, r_values[1])
        img_strong3, img_B_strong3 = swapFrequencies_3Channel(img_weak, img_B_weak, r_values[2])
        img_strong4, img_B_strong4 = swapFrequencies_3Channel(img_weak, img_B_weak, r_values[3])
        
        [img_strong, img_B_strong] = self.augm.transform_strong([img_strong, img_B_strong],[])
        [img_strong2, img_B_strong2] = self.augm.transform_strong([img_strong2, img_B_strong2],[])
        [img_strong3, img_B_strong3] = self.augm.transform_strong([img_strong3, img_B_strong3],[])
        [img_strong4, img_B_strong4] = self.augm.transform_strong([img_strong4, img_B_strong4],[])


        input_rgb = label.permute(0, 3, 1, 2)  # 将通道维度调整到正确的位置
        B, C, H, W = input_rgb.shape
        input_rgb = input_rgb.view(B, C, -1)  # 展平 H 和 W 维度

        # 将 RGB 像素值转换为类别索引
        class_indices = torch.zeros(B, H*W, dtype=torch.int64)

        img_aug = torch.stack([img_strong, img_strong2, img_strong3, img_strong4, img_weak, img_B_weak, img_B_strong4, img_B_strong3, img_B_strong2, img_B_strong], dim=1)

        for color, class_idx in color_to_class.items():
            mask = torch.all(input_rgb == torch.tensor(color).unsqueeze(0).unsqueeze(2), dim=1)
            class_indices[mask] = class_idx
        # 将类别索引重塑回图像形状
        output = class_indices.view(B, H, W)
        
        return {'name': name, 'A': img_weak, 'B': img_B_weak, 'L': output, 'L_binary': label_binary, 'aug': img_aug, 'compute_semantic': compute_semantic} # 'cls':class_label, 

class CDDataset_binary(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True, data_name=None):
        super(CDDataset_binary, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        L_path_binary = get_binary_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        
        label_binary = np.array(Image.open(L_path_binary), dtype=np.uint8)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label_binary = label_binary // 255

        output = False
        compute_semantic = False

        r_values = [4, 8, 16, 32]

        [img_weak, img_B_weak], [label_binary] = self.augm.transform([img, img_B],[label_binary], to_tensor=self.to_tensor)
        
        img_strong, img_B_strong = swapFrequencies_3Channel(img_weak, img_B_weak, r_values[0])
        img_strong2, img_B_strong2 = swapFrequencies_3Channel(img_weak, img_B_weak, r_values[1])
        img_strong3, img_B_strong3 = swapFrequencies_3Channel(img_weak, img_B_weak, r_values[2])
        img_strong4, img_B_strong4 = swapFrequencies_3Channel(img_weak, img_B_weak, r_values[3])
        
        
        [img_strong, img_B_strong] = self.augm.transform_strong([img_strong, img_B_strong],[])
        [img_strong2, img_B_strong2] = self.augm.transform_strong([img_strong2, img_B_strong2],[])
        [img_strong3, img_B_strong3] = self.augm.transform_strong([img_strong3, img_B_strong3],[])
        [img_strong4, img_B_strong4] = self.augm.transform_strong([img_strong4, img_B_strong4],[])


        img_aug = torch.stack([img_strong, img_strong2, img_strong3, img_strong4, img_weak, img_B_weak, img_B_strong4, img_B_strong3, img_B_strong2, img_B_strong], dim=1)
        
        return {'name': name, 'A': img_weak, 'B': img_B_weak, 'L': label_binary, 'L_binary': label_binary, 'aug': img_aug, 'compute_semantic': compute_semantic}
