import os
import numpy as np
import shutil

def load_ground_truth(gt_folder, image_name):
    """读取真值文件"""
    gt_file = os.path.join(gt_folder, f"{image_name}.txt")
    gt_boxes = []
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file {gt_file} not found.")
    # print(f"Reading ground truth file: {gt_file}")
    with open(gt_file, 'r') as f:
        for line in f.readlines():
            # print(f"Ground truth line: {line.strip()}")
            data = line.strip().split()
            # 真值格式：类别 cx cy w h
            cls, cx, cy, w, h = map(float, data)
            gt_boxes.append([cx, cy, w, h, int(cls)])
    return gt_boxes

def load_predictions(pred_folder, image_name):
    """加载预测结果"""
    pred_file = os.path.join(pred_folder, f"{image_name}.txt")
    pred_boxes = []
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file {pred_file} not found.")
    # print(f"Reading prediction file: {pred_file}")
    with open(pred_file, 'r') as f:
        for line in f.readlines():
            # print(f"Prediction line: {line.strip()}")
            data = line.strip().split()
            # 预测格式：类别 cx cy w h conf
            cls, cx, cy, w, h, conf = map(float, data)
            pred_boxes.append([cx, cy, w, h, conf, int(cls)])
    return pred_boxes

def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def compute_ap(gt_folder, pred_folder, iou_threshold=0.5):
    """计算平均精度（AP）"""
    all_gt_boxes = []
    all_pred_boxes = []
    ap_per_image = []  # 用于记录每张图像的 AP

    for file_name in os.listdir(pred_folder):  # 遍历预测文件
        if not file_name.endswith('.txt'):
            continue

        image_name = file_name.replace('.txt', '')
        gt_file = os.path.join(gt_folder, f"{image_name}.txt")
        pred_file = os.path.join(pred_folder, file_name)

        # 检查真实标签文件是否存在
        if not os.path.exists(gt_file):
            print(f"Ground truth file for {image_name} not found. Setting AP to 0.")
            ap_per_image.append(0)  # 直接记录 AP 为 0
            continue

        gt_boxes = load_ground_truth(gt_folder, image_name)
        pred_boxes = load_predictions(pred_folder, image_name)

        all_gt_boxes.extend(gt_boxes)
        all_pred_boxes.extend(pred_boxes)

        if not pred_boxes:  # 若预测为空，AP 为 0
            print(f"No predictions for {image_name}. Setting AP to 0.")
            ap_per_image.append(0)
            continue

        # 对预测框按置信度降序排序
        pred_boxes.sort(key=lambda x: x[4], reverse=True)

        tp = np.zeros(len(pred_boxes))  # True positives
        fp = np.zeros(len(pred_boxes))  # False positives
        used_gt = set()

        for i, pred_box in enumerate(pred_boxes):
            pred_cls = pred_box[5]
            best_iou = 0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                if j in used_gt or gt_box[4] != pred_cls:
                    continue
                iou = calculate_iou(pred_box[:4], gt_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp[i] = 1
                used_gt.add(best_gt_idx)
            else:
                fp[i] = 1

        # 计算 Precision 和 Recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-16)
        recall = cum_tp / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros(len(tp))

        # 计算 AP
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        ap_per_image.append(ap)

    # 计算所有图像的平均 AP
    mean_ap = np.mean(ap_per_image) if ap_per_image else 0
    return mean_ap, precision, recall


def organize_files_by_class(gt_folder, pred_folder):
    """
    将 gt_folder 和 pred_folder 中的文件按类别分类，并在各自目录内生成对应类别的新文件夹。
    
    Args:
        gt_folder (str): 真值文件夹路径。
        pred_folder (str): 预测结果文件夹路径。
    """
    # 获取文件列表
    gt_files = os.listdir(gt_folder)
    pred_files = os.listdir(pred_folder)

    # 提取类别
    gt_classes = set(f.split('_')[0] for f in gt_files if '_' in f)
    pred_classes = set(f.split('_')[0] for f in pred_files if '_' in f)
    all_classes = gt_classes.union(pred_classes)

    # 按类别分类文件并存储到对应文件夹
    for class_name in all_classes:
        # 创建类别文件夹
        class_gt_folder = os.path.join(gt_folder, class_name)
        class_pred_folder = os.path.join(pred_folder, class_name)
        os.makedirs(class_gt_folder, exist_ok=True)
        os.makedirs(class_pred_folder, exist_ok=True)

        # 将属于该类别的文件复制到对应文件夹
        for file in gt_files:
            if file.startswith(class_name + '_'):
                shutil.copy(os.path.join(gt_folder, file), class_gt_folder)
        for file in pred_files:
            if file.startswith(class_name + '_'):
                shutil.copy(os.path.join(pred_folder, file), class_pred_folder)

    print("分类完成，文件已存储到各自的类别文件夹中。")


def compute_classwise_ap(gt_base_folder, pred_base_folder, iou_threshold=0.5):
    """
    遍历 gt_folder 和 pred_folder 的所有子文件夹（类别文件夹），计算每一类的 AP、Precision 和 Recall。
    
    Args:
        gt_base_folder (str): 真值文件夹的路径，包含按类别组织的子文件夹。
        pred_base_folder (str): 预测文件夹的路径，包含按类别组织的子文件夹。
        iou_threshold (float): 计算 IoU 的阈值。
    
    Returns:
        dict: 包含每类的 AP、Precision 和 Recall 的字典，并计算 mAP。
    """
    # 获取所有类别（子文件夹名）
    categories = set(os.listdir(gt_base_folder)).intersection(os.listdir(pred_base_folder))

    # 存储结果
    results = {}
    total_ap = 0
    num_categories = 0

    # 遍历每个类别
    for category in categories:
        gt_folder = os.path.join(gt_base_folder, category)
        pred_folder = os.path.join(pred_base_folder, category)

        # 确保子文件夹存在
        if not os.path.isdir(gt_folder) or not os.path.isdir(pred_folder):
            continue

        # 调用 compute_ap 计算当前类别的 AP、Precision 和 Recall
        ap, precision, recall = compute_ap(gt_folder, pred_folder, iou_threshold=iou_threshold)

        # 保存结果
        results[category] = {
            "AP": ap,
            "Precision": precision,
            "Recall": recall
        }
        print(f"Category: {category}, AP: {ap}, Precision: {precision}, Recall: {recall}")

        # 累加 AP 计算 mAP
        total_ap += ap
        num_categories += 1

    # 计算 mAP
    mAP = total_ap / num_categories if num_categories > 0 else 0
    results['mAP'] = mAP
    print(f"Mean Average Precision (mAP): {mAP}")

    return results


# 输入路径
gt_folder = '/home/user/ccl/yolov9/data/NEU-DET/test_split/labels/'  # 真值文件夹路径
pred_folder = '/home/user/ccl/yolov9/runs/detect/yolov9_c_640_detect10/'  # 预测结果文件夹路径

# 调用函数进行分类
organize_files_by_class(gt_folder, pred_folder)

# 计算每类的 AP、Precision 和 Recall
results = compute_classwise_ap(gt_folder, pred_folder, iou_threshold=0.5)




