# -*- coding: utf-8 -*-
"""
功能：keras-yolov3 进行批量测试 并 保存结果
项目来源：https://github.com/qqwweee/keras-yolo3
"""

import colorsys
import os
from timeit import default_timer as timer
import time

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

path = '../mAP-master/input/images-optional' #待检测图片的位置

# 创建创建一个存储检测结果的dir
result_path = './result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# result如果之前存放的有文件，全部清除
for i in os.listdir(result_path):
    path_file = os.path.join(result_path,i)  
    if os.path.isfile(path_file):
        os.remove(path_file)

#创建一个记录检测结果的文件
txt_path =result_path + '/result.txt'
file = open(txt_path,'w')  
gpu_num = 1
class YOLO(object):
    def __init__(self):#由于这个是在ml_de中被调用的，所以下面的路径中的./是指ml_de.py所在的目录
        self.model_path = './logs/voc2012/trained_weights_final.h5' # model path or trained weights path
        #self.model_path = 'model_data/trained_weights_final.h5' # model path or trained weights path
        self.anchors_path = './model_data/yolo_anchors.txt'
        #self.classes_path = 'model_data/coco_classes.txt'
        self.classes_path = './model_data/voc_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()
        # print("**************************************************",self.scores)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
#        if gpu_num>=2:
#            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()  # 开始计时

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)  # 打印图片的尺寸
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 提示用于找到几个bbox

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * image.size[1] + 0.2).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        # # 保存框检测出的框的个数
        # file.write('find  ' + str(len(out_boxes)) + ' target(s) \n')

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # # 写入检测位置
            # file.write(
            #     predicted_class + '  score: ' + str(score) + ' \nlocation: top: ' + str(top) + '、 bottom: ' + str(
            #         bottom) + '、 left: ' + str(left) + '、 right: ' + str(right) + '\n')

            file.write(predicted_class + ' ' + str(score) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + ';')

            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print('time consume:%.3f s ' % (end - start))
        return image

    def close_session(self):
        self.sess.close()


# 图片检测

if __name__ == '__main__':

    t1 = time.time()
    yolo = YOLO()   
    for filename in os.listdir(path):        
        image_path = path+'/'+filename
        portion = os.path.split(image_path)
        # file.write(portion[1]+' detect_result：\n')
        file.write(portion[1] + ' ')  # 这里要有一个空格
        image = Image.open(image_path)
        r_image = yolo.detect_image(image)
        file.write('\n')
        # r_image.show() #显示检测结果
        # image_save_path = './result/result_'+portion[1]
        # print('detect result save to....:'+image_save_path)
        # r_image.save(image_save_path)

    time_sum = time.time() - t1
    # file.write('time sum: '+str(time_sum)+'s')
    print('time sum:',time_sum)
    file.close() 
    yolo.close_session()
