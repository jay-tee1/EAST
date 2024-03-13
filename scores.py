import os

from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


# define path to ground truth and to predicted data
INFERENCE_PATH = './outputs/confidences'
GROUND_TRUTH_PATH = './test_data/test_images_notmixed'




# load predicted boxes to tupels o coordinates and the confidence score
def load_boxes(path):
    boxes = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.isspace():  # not empty row
                    values = stripped_line.split(',')[:9]
                    score = float(values[8])    # confidence for detected box
                    box = list(map(int, values[:8]))
                    box = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
                    boxes.append((box, score))
    except FileNotFoundError:
        return []
    return boxes

# load boxes from ground truth to 4 tupels
def load_truth(path):
    boxes = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.isspace():  # check if row is not empty
                    values = stripped_line.split(',')[:8]
                    box = list(map(int, values))
                    box = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
                    boxes.append(box)
    except FileNotFoundError:
        return []
    return boxes





def calc(image_name, thresh):
    gt_boxes = load_truth(f"{GROUND_TRUTH_PATH}/{image_name}")
    pred_boxes = load_boxes(f"{INFERENCE_PATH}/{image_name}")

    # ensure that only one predicted box can be a true positive for a ground truth bounding box
    already_used_boxes = []

    fp = 0
    tp = 0

    # if no boxes were predicted for a ground truth image
    if not pred_boxes:
        return 0, 0, len(gt_boxes), []
    

    result = []
    for pred_box,score in pred_boxes:
        ious = []

        for gt_box in gt_boxes:
            polygon1 = Polygon(gt_box)
            polygon2 = Polygon(pred_box)

            try:
                # compute IoU
                intersect = polygon1.intersection(polygon2).area
                union = polygon1.union(polygon2).area

                iou = intersect / union
                ious.append(iou)
            except Exception as e:
                pass

        # check if IoU exeeds threshold
        if np.max(ious) < thresh:
            result.append((score, 0, 1))
            fp += 1
        else:
            index = np.argmax(ious)

            if index in already_used_boxes:
                result.append((score, 0, 1))
                fp += 1
                continue
            else:
                result.append((score, 1, 0))
                tp += 1

    # return amount of true positives and false positives, as well as amount of ground truth bounding boxes and tuple of confidence + whether tp or fp
    return tp, fp, len(gt_boxes), result



#plots the precision x recall curve and the interpolated curve
def plot_precision_recall_curve(precision_recall_tuples):
    precisions, recalls = zip(*precision_recall_tuples)

    plt.plot(recalls, precisions, marker=',', linestyle='solid', color='b', label='Precision x Recall Curve')

    decreasing_max_precision = np.maximum.accumulate(precisions[::-1])[::-1]
    plt.step(recalls, decreasing_max_precision, linestyle='dashed', color='r', label='Interpolated Curve')
    area = auc(recalls, decreasing_max_precision)
    area2 = auc(recalls, precisions)


    plt.title('Precision x Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(False)
    plt.legend()
    plt.show()
    return area2


if __name__ == '__main__':
    # get all text files in the ground truth path
    files = [f for f in os.listdir(GROUND_TRUTH_PATH) if f.endswith('.txt')]

    total_tp = 0
    total_fp = 0
    total_sum = 0

    total_result = []
    # IoU thresholds used for mAP
    thresh = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    i = 0
    area = 0
    # calcualte mAP
    for t in thresh:
        threshold = thresh[i]
        i += 1
        for file in files:
            tp, fp, sum, result = calc(file, threshold)
            total_tp += tp
            total_fp += fp
            total_sum += sum
            total_result.extend(result)

        total_result.sort(key=lambda x: x[0], reverse=True)


        acc_tp = 0
        acc_fp = 0
        curve = []
        for result in total_result:
            if result[1] == 1:
                acc_tp += 1
            elif result[2] == 1:
                acc_fp += 1
            rec = acc_tp/ total_sum
            pre = acc_tp/(acc_tp + acc_fp)
            curve.append((pre, rec))
        #print(curve[0], curve[1], curve[2], curve[4], curve[5])

        area += plot_precision_recall_curve(curve)

        # corresponding precision, recall and f1 scores for a IoU threshold
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / total_sum
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
    
    mAP = area/10
    print("mAP   ", mAP)


    
