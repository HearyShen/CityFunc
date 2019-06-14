import os
import random
import shutil

dataset_root = r"D:\Datasets\CityFunc"
train_image_root = os.path.join(dataset_root, "train", "image")
train_visit_root = os.path.join(dataset_root, "train", "visit")
val_image_root = os.path.join(dataset_root, "val", "image")
val_visit_root = os.path.join(dataset_root, "val", "visit")
test_image_root = os.path.join(dataset_root, "test", "image")
test_visit_root = os.path.join(dataset_root, "test", "visit")


os.makedirs(val_visit_root)
os.makedirs(test_visit_root)
val_count = 0
test_count = 0

for classdir in os.listdir(train_image_root):
    class_root = os.path.join(train_image_root, classdir)
    print("entering " + class_root)
    os.makedirs(os.path.join(val_image_root, classdir))
    os.makedirs(os.path.join(test_image_root, classdir))
    for img in os.listdir(class_root):
        img_name = os.path.splitext(img)[0]
        img_path = os.path.join(class_root, img)
        visit_path = os.path.join(train_visit_root, img_name+".txt")
        rand = random.random()
        if rand >= 0.8:
            if rand < 0.9:      # move to val set
                img_dest_path = os.path.join(val_image_root, classdir, img)
                visit_dest_path = os.path.join(val_visit_root, img_name+".txt")
                val_count += 1
            else:                   # move to test set
                img_dest_path = os.path.join(test_image_root, classdir, img)
                visit_dest_path = os.path.join(test_visit_root, img_name+".txt")
                test_count += 1
            
            shutil.move(img_path, img_dest_path)
            shutil.move(visit_path, visit_dest_path)

print("train/val/test: %d/%d/%d" % (40000 - val_count - test_count, val_count, test_count))