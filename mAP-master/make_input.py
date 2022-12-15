import os
import shutil

test_dir = '../data/voc2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
img_dir = '../data/voc2012/VOCdevkit/VOC2012/JPEGImages'
output_dir = './input/images-optional'

test_jpg = os.listdir(img_dir)
# print(test_jpg[0])

num = len(test_jpg)
list = range(num)

count = 0
file = open(test_dir, 'r', encoding='UTF-8')
for line in file:
        count = count+1

print(count)

count = 0
for i in list:
    name = test_jpg[i][:-4]
    # print("name:",name)
    file = open(test_dir, 'r', encoding='UTF-8')
    for line in file:
        # print(line)
        if(name.strip() == line.strip()):
            count = count+1
            shutil.copy(os.path.join(img_dir, name+'.jpg'), output_dir)
            break
print("finall:",count)
