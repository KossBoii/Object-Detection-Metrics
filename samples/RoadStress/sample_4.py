import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *
import argparse
import numpy as np
import cv2
import glob
import os
import csv
import numpy as np

def get_boxes(dir, allBoundingBoxes, dataset_type, bb_type):
    result = allBoundingBoxes.clone()

    for file in glob.iglob(dir + "/*.txt"):
        img_name = file[-12:-4]
        with open(file, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                vals = line.split(" ")
                assert(bb_type=="gt" or bb_type=="dt")

                if(bb_type == "gt"):
                    assert(len(vals) == 5)

                    idClass = vals[0] #class
                    x1 = float(vals[1]) 
                    y1 = float(vals[2])
                    x2 = float(vals[3])
                    y2 = float(vals[4])

                    if(dataset_type == "new"):
                        bb = BoundingBox(img_name, idClass, x1, y1, x2, y2, CoordinatesType.Absolute, (4864, 3648), BBType.GroundTruth, format=BBFormat.XYX2Y2)
                    elif(dataset_type == "old"):
                        bb = BoundingBox(img_name, idClass, x1, y1, x2, y2, CoordinatesType.Absolute, (4000, 3000), BBType.GroundTruth, format=BBFormat.XYX2Y2)
                elif(bb_type == "dt"):
                    assert(len(vals) == 6)

                    idClass = vals[0]  # class
                    confidence = float(vals[1])  # confidence
                    x1 = float(vals[2])
                    y1 = float(vals[3])
                    x2 = float(vals[4])
                    y2 = float(vals[5])

                    if(dataset_type == "new"):
                        bb = BoundingBox(img_name, idClass, x1, y1, x2, y2, CoordinatesType.Absolute, (4864, 3648), BBType.Detected, confidence, format=BBFormat.XYX2Y2)
                    elif(dataset_type == "old"):
                        bb = BoundingBox(img_name, idClass, x1, y1, x2, y2, CoordinatesType.Absolute, (4000, 3000), BBType.Detected, confidence, format=BBFormat.XYX2Y2)
                result.addBoundingBox(bb)
    return result

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 evaluation for road stress")
    parser.add_argument("--gt", required=True, help="path to groundtruth directory")
    parser.add_argument("--dt", required=True, help="path to detection directory")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Arguments: " + str(args))

    for d in ["old", "new"]:
        dataset_name = "roadstress_" + d + "_val"
        print("Dataset: " + dataset_name)

        # result_AP = []
        # result_AR = []
        result = []

        bbox_gt = BoundingBoxes()
        bbox_gt = get_boxes(args.gt + dataset_name, bbox_gt, dataset_type=d, bb_type="gt")
        for i in np.arange(0.05, 0.95, 0.05):
            i = i.round(decimals=2)
            print("Threshold: " + str(i))
            all_bbox = get_boxes(args.dt + dataset_name + "/threshold_%.2f" % i, 
            bbox_gt, dataset_type=d, bb_type="dt")

            save_path = "./plot/%s/threshold_%.2f/" % (args.dt[13:-1], i)
            os.makedirs(save_path, exist_ok=True)

            evaluator = Evaluator()
            # evaluator.PlotPrecisionRecallCurve(
            #     all_bbox,  # Object containing all bounding boxes (ground truths and detections)
            #     IOUThreshold=i,  # IOU threshold
            #     method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
            #     showAP=True,  # Show Average Precision in the title of the plot
            #     showInterpolatedPrecision=True,
		    #     savePath=save_path)  # Plot the interpolated precision curve
            
            # Get metrics with PASCAL VOC metrics
            metricsPerClass = evaluator.GetPascalVOCMetrics(
                all_bbox,  # Object containing all bounding boxes (ground truths and detections)
                IOUThreshold=i,  # IOU threshold
                method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
            print("Average precision values per class:")
            # Loop through classes to obtain their metrics
            for mc in metricsPerClass:
                # Get metric values per each class
                c = mc['class']
                precision = mc['precision']
                recall = mc['recall']
                average_precision = mc['AP']
                ipre = mc['interpolated precision']
                irec = mc['interpolated recall']
                total_tp = mc['total TP']
                total_fp = mc['total FP']
                total_pos = mc['total positives']

                # Print AP per class
                print("%s: %s" % ("precision", str(precision)))
                print("%s: %s" % ("recall", str(recall)))
                print("%s: %f" % ("AP", average_precision))
                print("%s: %d" % ("total TP", total_tp))
                print("%s: %d" % ("total FP", total_fp))
                print("%s: %d" % ("total Pos", total_pos))

                result.append(precision)
                result.append(recall)

                result_trans = np.array(result).T

                pth = "./output/%s/" % args.dt[13:-1]
                os.makedirs(pth, exist_ok=True)
                with open(pth + "threshold_%.2f_output.csv" % i, "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(result_trans)



