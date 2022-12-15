import os
import shutil

testfilepath='./input/images-optional'
xmlfilepath = '../data/voc2012/VOCdevkit/VOC2012/Annotations/'
xmlsavepath = './input/ground-truth/'
test_jpg = os.listdir(testfilepath)

num = len(test_jpg)
list = range(num)
L=[]
print(num)
for i in list:
    name = test_jpg[i][:-4] + '.xml'
    # print(name)
    L.append(name)
for filename in L:
    print(filename)
    shutil.copy(os.path.join(xmlfilepath,filename),xmlsavepath)
