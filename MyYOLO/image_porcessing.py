import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from models.yolo.util.deHaze import deHaze

dataset = 'train_1w'

print('h')
# 水平翻转
with open('./data/car/image/' + dataset + '/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
    src_file.readline()
    with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_h.csv', 'w') as porc_file:
        for line in src_file:
            line = line.strip('\n')
            label_list = line.split(',')
            image_name = label_list[0]
            position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

            write_line = 'h_' + image_name + ','
            for position in position_list:
                if position[0] == '':
                    write_line += ';'
                    break

                w = float(position[2])
                h = float(position[3])
                x = 1069.0 - float(position[0]) - w
                y = float(position[1])
                c = float(position[4])

                write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

            image_path = os.path.join('data', 'car', 'image', dataset, image_name)
            target_image_path = os.path.join('data', 'car', 'image', dataset, 'h_' + image_name)
            image = cv2.imread(image_path)
            image = cv2.flip(image, 1)
            cv2.imwrite(target_image_path, image)

            porc_file.write(write_line[:-1] + '\n')

with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
    src_file.write('\n')
    with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_h.csv', 'r') as porc_file:
        for line in porc_file:
            line = line.split(',')
            img_path = os.path.join('data', 'car', 'image', dataset, line[0])
            line = img_path + ',' + line[1]
            src_file.write(line)

# print('v')
# # 上下翻转
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     src_file.readline()
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_v.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'v_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = 500 - float(position[1]) - h
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'v_' + image_name)
#             image = cv2.imread(image_path)
#             image = cv2.flip(image, 0)
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')

# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_v.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)

# print('brightness up')
# # 提高亮度
# seq = iaa.Sequential(
#     [
#         iaa.Multiply(mul=1.4, per_channel=False, name=None, deterministic=False, random_state=None),
#     ]
# )
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_brightness_up.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'bu_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = float(position[1])
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'bu_' + image_name)
#             image = cv2.imread(image_path)
#             image = seq.augment_image(image)
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')


# print('brightness down')
# #降低亮度
# seq = iaa.Sequential(
#     [
#         iaa.Multiply(mul=0.6, per_channel=False, name=None, deterministic=False, random_state=None),
#     ]
# )
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_brightness_down.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'bd_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = float(position[1])
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'bd_' + image_name)
#             image = cv2.imread(image_path)
#             image = seq.augment_image(image)
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')

# # 增加曝光度, gamma函数处理
# def gamma_trans(img,gamma):
#     gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)] #建立映射表
#     gamma_table=np.round(np.array(gamma_table)).astype(np.uint8) #颜色值为整数
#     return cv2.LUT(img,gamma_table) #图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法

# print('gamma up')
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_gamma_up.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'gu_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = float(position[1])
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'gu_' + image_name)
#             image = cv2.imread(image_path)
#             image = gamma_trans(image, 0.7)
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')

# print('gamma down')
# # 降低曝光度
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_gamma_down.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'gd_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = float(position[1])
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'gd_' + image_name)
#             image = cv2.imread(image_path)
#             image = gamma_trans(image, 1.3)
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')

# # 高斯模糊
# seq = iaa.Sequential(
#     [
#         iaa.GaussianBlur(2.1)
#     ]
# )
# print('gauss')
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     src_file.readline()
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_gauss.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'gauss_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = float(position[1])
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'gauss_' + image_name)
#             image = cv2.imread(image_path)
#             image = seq.augment_images(image)
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')

# # 中值模糊
# seq = iaa.Sequential(
#     [
#         iaa.MedianBlur(1.9)
#     ]
# )
# print('median')
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     src_file.readline()
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_median.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'median_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = float(position[1])
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'median_' + image_name)
#             image = cv2.imread(image_path)
#             image = seq.augment_images(image)
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')

# # deHaze
# print('deHaze')
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     src_file.readline()
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_deHaze.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'deHaze_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = float(position[1])
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'deHaze_' + image_name)
#             image = cv2.imread(image_path)
#             image = deHaze(image / 255) * 255
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')

# # sharpen
# seq = iaa.Sequential(
#     [
#         iaa.Sharpen(0.1)
#     ]
# )
# print('sharpen')
# with open('./data/car/train/' + dataset + '.csv', 'r', encoding='utf-8') as src_file:
#     src_file.readline()
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_sharpen.csv', 'w') as porc_file:
#         for line in src_file:
#             line = line.strip('\n')
#             label_list = line.split(',')
#             image_name = label_list[0].split(os.sep)[-1]
#             position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

#             write_line = 'sharpen_' + image_name + ','
#             for position in position_list:
#                 if position[0] == '':
#                     write_line += ';'
#                     break

#                 w = float(position[2])
#                 h = float(position[3])
#                 x = float(position[0])
#                 y = float(position[1])
#                 c = float(position[4])

#                 write_line += '{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(x, y, w, h, c) + ';'

#             image_path = os.path.join('data', 'car', 'image', dataset, image_name)
#             target_image_path = os.path.join('data', 'car', 'image', dataset, 'sharpen_' + image_name)
#             image = cv2.imread(image_path)
#             image = seq.augment_images(image)
#             cv2.imwrite(target_image_path, image)

#             porc_file.write(write_line[:-1] + '\n')

# # 提高亮度
# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_brightness_up.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)

# # 降低亮度
# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_brightness_down.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)

# # 增加曝光度
# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_gamma_up.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)

# # 降低曝光度
# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_gamma_down.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)

# # 高斯模糊
# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_gauss.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)

# # 中值模糊
# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_median.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)

# # deHaze
# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_deHaze.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)

# # sharpen
# with open('./data/car/train/' + dataset + '.csv', 'a+', encoding='utf-8') as src_file:
#     with open('./data/car/image/' + dataset + '/' + dataset + '_image_porcess_sharpen.csv', 'r') as porc_file:
#         for line in porc_file:
#             line = line.split(',')
#             img_path = os.path.join('data', 'car', 'image', dataset, line[0])
#             line = img_path + ',' + line[1]
#             src_file.write(line)