import os
from PIL import Image, ImageDraw
import torch
from PIL import ImageFile
import csv
import  xml.dom.minidom
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    print(tf.cigure.is_gpu_available())
    print(tf.test.is_built_with_cuda())