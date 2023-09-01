from random import sample, shuffle

from numpy import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data.dataset import Dataset
import time

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, \
                 mosaic, mixup, meter_aug, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.meter_au = meter_aug

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])  #####包含了4个元素，3个来自随机，其他1个来自指定的数据。
            shuffle(lines)
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)



        if self.meter_au:
            lines = sample(self.annotation_lines, 2)
            lines.append(self.annotation_lines[index])  #####包含了3个元素，2个来自随机，其他1个来自指定的
            shuffle(lines)

            image, box = self.meter_aug(lines)

            if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[index])  #####包含了4个元素，3个来自随机，其他1个来自指定的数据。
                shuffle(lines)
                image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)




        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape  ###416，，，640...
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # ---------------------------------#
            #   每一行进行分割
            # ---------------------------------#
            line_content = line.split()
            # ---------------------------------#
            #   打开图片
            # ---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # ---------------------------------#
            #   图片的大小
            # ---------------------------------#
            iw, ih = image.size
            # ---------------------------------#
            #   保存框的位置
            # ---------------------------------#
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # ---------------------------------#
            #   是否翻转图片
            # ---------------------------------#
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # ------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            # ------------------------------------------#
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # -----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            # -----------------------------------------------#
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # ---------------------------------#
            #   对box进行重新处理
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # ---------------------------------#
        #   将图片分割，放在一起
        # ---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对框进行进一步的处理
        # ---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes

    def meter_aug(self, lines, hue=.1, sat=0.7, val=0.4):

        pc1 = lines[0]  # 在3张图片中选第一张
        line_content1 = pc1.split()
        img1 = Image.open(line_content1[0])
        img1 = cvtColor(img1)

        iw, ih = img1.size
        box1 = np.array([np.array(list(map(int, box.split(',')))) for box in line_content1[1:]])

        new_box1 = self.box_target(box1)

        max_height_box1 = np.max(new_box1[:, 3] - new_box1[:, 1])
        max_width_box1 = np.max(new_box1[:, 2] - new_box1[:, 0])


        for lin in lines:

            line_content = lin.split()
            image = Image.open(line_content[0])
            image = cvtColor(image)  # 打开图片，开始抠图

            # ------------------------------- -----
            #        需要对抠图的图片做进一步的处理
            # --------------------------------- --
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])
            new_bo = self.box_target(box)

            try:
                max_height_box = np.max(new_bo[:, 3] - new_bo[:, 1])
                max_width_box = np.max(new_bo[:, 2] - new_bo[:, 0])
            except:
                print(new_bo)
                print(len(new_bo))
                print(line_content[0])
                bo = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])
                new = self.box_target(bo)
                max_height_box = np.max(new[:, 3] - new[:, 1])
                max_width_box = np.max(new[:, 2] - new[:, 0])

            # -------------与pc1中的进行比较大小，如果目标太大那么就需要对图片进行缩小，然后再进行抠图
            we_rato = float(max_height_box1) / max_height_box
            he_rato = float(max_width_box1) / max_width_box

            if we_rato > he_rato:
                m = we_rato
            else:
                m = he_rato

            # ------------对需要抠图的图片进行放缩
            image = image.resize((int(image.width * m), int(image.height * m)), Image.BICUBIC)
            box[:, [0, 2]] = (box[:, [0, 2]] * m).astype(int)
            box[:, [1, 3]] = (box[:, [1, 3]] * m).astype(int)

            new_box2 = self.box_target(box)

            for i in range(len(new_box2)):
                width = new_box2[i][2] - new_box2[i][0]
                height = new_box2[i][3] - new_box2[i][1]
                c_x = int(random.uniform(width, iw - width))
                c_y = int(random.uniform(height, ih - height))

                # --------------------对框进行复制------------------------------
                box2 = new_box2[i].copy()
                box2[0] = c_x - (width // 2)
                box2[1] = c_y - (height // 2)
                box2[2] = c_x + (width // 2)
                box2[3] = c_y + (height // 2)

                # --------裁剪目标区域------------
                img2 = image.crop((new_box2[i][0], new_box2[i][1], new_box2[i][2], new_box2[i][3]))

                # -------判断iou并重新更新中心点---------
                temiou = 0
                flag = 0
                for m in range(len(box1)):
                    iou = self.calculate_iou(box1[m], box2)

                    if iou > 0.1:
                        temiou = iou
                if temiou > 0.1:
                    flag = 1
                co=0
                while (flag):
                    co+=1

                    c_x = int(random.uniform(width, iw - width))
                    c_y = int(random.uniform(height, ih - height))
                    box2[0] = c_x - (width // 2)
                    box2[1] = c_y - (height // 2)
                    box2[2] = c_x + (width // 2)
                    box2[3] = c_y + (height // 2)

                    temiou2 = 0
                    for m in range(len(box1)):
                        iou = self.calculate_iou(box1[m], box2)
                        if iou > 0.1:
                            temiou2 = iou
                    if temiou2 < 0.1:
                        flag = 0
                    if co==20:
                        break

                # -------将裁剪的图片进行复制---------

                r = random.uniform(0.6, 1)

                img2 = img2.convert("RGBA")
                alpha = img2.split()[3]
                enhancer = ImageEnhance.Brightness(alpha)
                enhanced_alpha = enhancer.enhance(r)

                img2 = Image.merge("RGBA", img2.split()[:3] + (enhanced_alpha,))

                img1.paste(img2, (box2[0], box2[1]))

                # -------对box进行操作---------
                box2 = np.expand_dims(box2, axis=0)
                box1 = np.concatenate((box1, box2), axis=0)

        # ------将图片尺寸进行缩放，使得图片的大小变为416或者640

        w_r = float(416.0) / img1.width
        h_r = float(416.0) / img1.height

        img1 = np.array(img1)
        img1 = cv2.resize(img1, (416, 416))
        new_image = np.array(img1, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))

        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        box1 = box1.astype(float)
        box1[:, [0, 2]] = (box1[:, [0, 2]] * w_r).astype(int)
        box1[:, [1, 3]] = (box1[:, [1, 3]] * h_r).astype(int)

        return new_image, box1

    def box_target(self,box):
        new_box = []
        for tmbox in box:
            if tmbox[4] != 2:
                new_box.append(tmbox)
        new_box=np.array(new_box)
        return new_box


    def calculate_iou(self, box1, box2):
        """
        计算两个框之间的 Intersection over Union（IoU）
        参数：
            box1：第一个框的坐标，格式为 (x1, y1, x2, y2)
            box2：第二个框的坐标，格式为 (x1, y1, x2, y2)
        返回值：
            iou：两个框之间的 IoU 值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        iou = intersection / union if union > 0 else 0
        return iou


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes










