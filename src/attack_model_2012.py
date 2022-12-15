import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from attacks.ml_lba_2007 import MLLBA
from attacks.ml_ba_2007 import MLBA
from yolo_master.yolo_voc2007 import *
import numpy as np
import os
from tqdm import tqdm

tqdm.monitor_interval = 0
class AttackModel():
	def __init__(self, state):
		self.state = state
		self.y_target = state['y_target']
		self.y = state['y']
		self.data_loader = tqdm(state['data_loader'], desc='ADV')
		self.ori_loader = state['ori_loader']
		self.model = state['model']
		self.adv_save_x = state['adv_save_x']
		self.adv_batch_size = state['adv_batch_size']
		self.adv_begin_step = state['adv_begin_step']
		self.save_image_folder = state['save_image_folder']
		self.attack_model = None
		self.yolonet =  YOLO()

	def attack(self):
		clip_min = 0.
		clip_max = 1.
		if self.state['adv_method'] == 'ml_lba':
			self.attack_model = MLLBA(self.model)
			params = {'pop_size': 100,
					  'generation':200,
					  'clip_min': clip_min,
					  'clip_max': clip_max,
					  'alpha':5,
					  'batch_size': self.adv_batch_size,
					  'yolonet': self.yolonet}
			self.ml_lba(params)
		elif self.state['adv_method'] == 'ml_ba':
			self.attack_model = MLBA(self.model)
			params = {'pop_size': 100,
					  'generation':200,
					  'clip_min': clip_min,
					  'clip_max': clip_max,
					  'batch_size': self.adv_batch_size,
					  'yolonet': self.yolonet,
					  'save_image_folder':self.save_image_folder}
			self.ml_ba(params)
		else:
			print('please choose a correct adv method')

	def ml_lba(self, params):
		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		print(params)
		for i, (input, target) in enumerate(self.data_loader):
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]
			adv ,count = self.attack_model.generate_np(input[0].cpu().numpy(),self.ori_loader, **params)
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)

	def ml_ba(self, params):
		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		print(params)
		for i, (input, target) in enumerate(self.data_loader):
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]
			params['image_index'] = 1 + i * batch_size;
			self.attack_model.generate_np(input[0].cpu().numpy(),self.ori_loader, **params)

def new_folder(file_path):
	folder_path = os.path.dirname(file_path)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def get_target_set(y, y_target):
	y[y == 0] = -1
	A_pos = np.logical_and(np.not_equal(y, y_target), y == 1) + 0
	A_neg = np.logical_and(np.not_equal(y, y_target), y == -1) + 0
	B_pos = np.logical_and(np.equal(y, y_target), y == 1) + 0
	B_neg = np.logical_and(np.equal(y, y_target), y == -1) + 0

	y_tor = A_pos * -2 + -1 * B_neg + 1 * B_pos + 2 * A_neg
	return y_tor, A_pos, A_neg, B_pos, B_neg
