import os
import shutil
import numpy as np

os.makedirs("./final_result/", exist_ok=True)

for model_name in os.listdir("./roadstress_new_val"):
    base_path = "./final_result/%s" % model_name
    os.makedirs(base_path, exist_ok=True)

    for i in np.arange(0.05, 0.95, 0.05):
        for d in ["new", "old"]:
            src = "./roadstress_%s_val/%s/threshold_%.2f/roadstress.png" % (d, model_name, i)
            des = (base_path + "/threshold_%.2f_%s.png") % (i, d)
            shutil.move(src, des)



    