import os
import random
from shutil import copy2

# 数据集路径
data_dir = '/home/tinnel/ccl/yolov9/data/NEU-DET/train'  # 原始 test 文件夹路径
val_dir = '/home/tinnel/ccl/yolov9/data/NEU-DET/s_train_split'    # 验证集路径
test_dir = '/home/tinnel/ccl/yolov9/data/NEU-DET/Uns_train_split'  # 重新划分的测试集路径

# 创建目标文件夹
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'annatation'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'annatation'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

# 六种类别
categories = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# 遍历每个类别并进行划分
for category in categories:
    # 获取当前类别的所有图片
    images = [f for f in os.listdir(os.path.join(data_dir, 'images')) if f.startswith(category)]
    random.shuffle(images)  # 打乱顺序以确保随机性

    # 计算划分数量
    total_images = len(images)
    val_count = int(total_images * 0.2)
    test_count = total_images - val_count

    val_images = images[:val_count]
    test_images = images[val_count:]

    # 将图片和对应的标注文件移动到目标文件夹
    for image in val_images:
        base_name = os.path.splitext(image)[0]
        # 复制图片
        copy2(os.path.join(data_dir, 'images', image), os.path.join(val_dir, 'images', image))
        # 复制 .xml 文件
        copy2(os.path.join(data_dir, 'annatation', base_name + '.xml'), os.path.join(val_dir, 'annatation', base_name + '.xml'))
        # 复制 .txt 文件
        copy2(os.path.join(data_dir, 'labels', base_name + '.txt'), os.path.join(val_dir, 'labels', base_name + '.txt'))

    for image in test_images:
        base_name = os.path.splitext(image)[0]
        # 复制图片
        copy2(os.path.join(data_dir, 'images', image), os.path.join(test_dir, 'images', image))
        # 复制 .xml 文件
        copy2(os.path.join(data_dir, 'annatation', base_name + '.xml'), os.path.join(test_dir, 'annatation', base_name + '.xml'))
        # 复制 .txt 文件
        copy2(os.path.join(data_dir, 'labels', base_name + '.txt'), os.path.join(test_dir, 'labels', base_name + '.txt'))

print("数据集划分完成！")
