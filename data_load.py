from glob import glob

class_whitelist = [10, 17, 47, 49, 55, 118, 120, 129, 130, 162, 171, 196] 

with open('CUB_200_2011/images.txt') as f:
    paths = [None] + [line.split()[1].split('/')[-1] for line in f.readlines()]
with open('CUB_200_2011/image_class_labels.txt') as f:
    classes = [None] + [int(line.split()[1]) for line in f.readlines()]
with open('CUB_200_2011/bounding_boxes.txt') as f:
    bounding_boxes = [None] + [tuple(map(float, line.split()[1:])) for line in f.readlines()]
with open('CUB_200_2011/train_test_split.txt') as f:
    lines = f.readlines()
    train_img_id = [int(line.split()[0]) for line in lines if line.split()[1] == '1']
    test_img_id = [int(line.split()[0]) for line in lines if line.split()[1] == '0']

train_img_id = [img_id for img_id in train_img_id if classes[img_id] in class_whitelist]
test_img_id = [img_id for img_id in test_img_id if classes[img_id] in class_whitelist]

with open('species_train.txt', 'w') as f:
    for img_id in train_img_id:
        print(paths[img_id], classes[img_id], file=f)
with open('species_test.txt', 'w') as f:
    for img_id in test_img_id:
        print(paths[img_id], classes[img_id], file=f)

with open('bbox_train.txt', 'w') as f:
    for img_id in train_img_id:
        print(paths[img_id], bounding_boxes[img_id], file=f)
with open('bbox_test.txt', 'w') as f:
    for img_id in test_img_id:
        print(paths[img_id], bounding_boxes[img_id], file=f)

with open('spec_bbox_train.txt', 'w') as f:
    for img_id in train_img_id:
        print(paths[img_id], classes[img_id], bounding_boxes[img_id], file=f)
with open('spec_bbox_test.txt', 'w') as f:
    for img_id in test_img_id:
        print(paths[img_id], classes[img_id], bounding_boxes[img_id], file=f)


    # image_paths = [line.split()[1] for line in f.readlines() if int(line.split()[0]) in class_whitelist]

