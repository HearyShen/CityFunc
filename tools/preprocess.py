import os

dataset_root = r"D:\Datasets\CityFunc"
train_image_root = os.path.join(dataset_root, "train", "image")
train_visit_root = os.path.join(dataset_root, "train", "visit")
val_image_root = os.path.join(dataset_root, "val", "image")
val_visit_root = os.path.join(dataset_root, "val", "visit")
test_image_root = os.path.join(dataset_root, "test", "image")
test_visit_root = os.path.join(dataset_root, "test", "visit")

img_with_visit = set()
img_without_visit = set()

for classdir in os.listdir(train_image_root):
    class_root = os.path.join(train_image_root, classdir)
    print("entering " + class_root)
    for img in os.listdir(class_root):
        img_name = os.path.splitext(img)[0]
        if os.path.exists(os.path.join(train_visit_root, img_name+".txt")):
            img_with_visit.add(img_name)
        else:
            img_without_visit.add(img_name)

print("Images with visit text:")
print("Total: " + str(len(img_with_visit)))
# print(sorted(img_with_visit))

print("Images without visit text:")
print("Total: " + str(len(img_without_visit)))
