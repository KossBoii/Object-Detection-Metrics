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

import matplotlib.pyplot as plt


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

        bbox_gt = BoundingBoxes()
        bbox_gt = get_boxes(args.gt + dataset_name, bbox_gt, dataset_type=d, bb_type="gt")
        
        pth = "./output/%s/" % args.dt[13:-1]
        os.makedirs(pth, exist_ok=True)
        
        for i in np.arange(0.05, 0.95, 0.05):
            i = i.round(decimals=2)
            print("Threshold: " + str(i))
            all_bbox = get_boxes(args.dt + dataset_name + "/threshold_%.2f" % i, 
            bbox_gt, dataset_type=d, bb_type="dt")

            evaluator = Evaluator()
            
            results = evaluator.GetPascalVOCMetrics(
                all_bbox,
                IOUThreshold=i,
                method=MethodAveragePrecision.EveryPointInterpolation
            )

            for result in results:
                if(len(results) == 1):
                    print("There is only 1 class")
                else:
                    print("There are more than 1 class")

                if result is None:
                    raise IOError('Error: Class %d could not be found.' % classId)

                classId = result['class']
                precision = result['precision']
                recall = result['recall']
                average_precision = result['AP']
                mpre = result['interpolated precision']
                mrec = result['interpolated recall']
                npos = result['total positives']
                total_tp = result['total TP']
                total_fp = result['total FP']

                print("TP: %d" % total_tp)
                print("FP: %d" % total_fp)

                plt.plot(recall, precision, label='%.2f' % i, linewidth=0.75)
    
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Precision-Recall curve for %s' % str(classId))
        plt.legend(title="IOU Threshold", bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()        
        plt.grid()
        plt.show()
        plt.savefig(os.path.join(pth, dataset_name + '.png'), dpi=250)
        plt.close()


