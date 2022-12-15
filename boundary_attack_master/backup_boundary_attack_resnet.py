from __future__ import print_function

try:
	raw_input
except:
	raw_input = input
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import sys
sys.path.append('../')
import json
import torch
import numpy as np
import time

from PIL import Image
from ml_liw_model.models import inceptionv3_attack
from ml_liw_model.voc import Voc2007Classification
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


RESNET_MEAN = np.array([0.4076, 0.4579, 0.4850])


def orthogonal_perturbation(delta, prev_sample, target_sample):
	"""Generate orthogonal perturbation."""
	perturb = np.random.randn(1, 448, 448, 3)
	perturb /= np.linalg.norm(perturb, axis=(1,2))
	perturb *= delta * np.mean(get_diff(target_sample, prev_sample))

	# Project perturbation onto sphere around target 将扰动投射到目标周围的球体上
	diff = (target_sample.permute(0, 2,3,1) - prev_sample.permute(0, 2,3,1)).cpu() # Orthorgonal vector to sphere surface 正交向量到球面
	diff /= get_diff(target_sample, prev_sample) # Orthogonal unit vector 正交单位向量

	# We project onto the orthogonal then subtract from perturb 我们投影到正交上，然后从扰动中减去
	# to get projection onto sphere surface 投影到球面上
	perturb = torch.from_numpy(perturb).float()
	perturb= perturb.permute(0, 3, 1, 2)
	# print(perturb.shape)
	# print(diff.shape)
	diff = diff.permute(0, 3, 1, 2)
	perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff)**2) * diff #**是幂运算
	# Check overflow and underflow
	perturb = np.clip(perturb, 0., 1.)
	return perturb.cuda()


def forward_perturbation(epsilon, prev_sample, target_sample):
	"""Generate forward perturbation."""
	# print(target_sample.shape)
	# print(prev_sample.shape)
	perturb = (target_sample - prev_sample)
	perturb *= epsilon
	return perturb.cuda()


def get_converted_prediction(sample, classifier):
	"""
	The original sample is dtype float32, but is converted
	to uint8 when exported as an image. The loss of precision
	often causes the label of the image to change, particularly
	because we are very close to the boundary of the two classes.
	This function checks for the label of the exported sample
	by simulating the export process.
	"""
	# sample = (sample + RESNET_MEAN).astype(np.uint8).astype(np.float32) - RESNET_MEAN
	# print(sample.shape)
	sample=torch.from_numpy(sample).float()
	sample= sample.permute(0, 3, 1, 2)
	sample=sample.cuda()
	# print(sample.shape)
	with torch.no_grad():
		label = classifier(sample)
	return label


def save_image(tar_iamge_name,sample, classifier, folder,attack_class,l2,isbest):
	"""Export image file."""
	with torch.no_grad():
		label = classifier(sample)
	label=label.cpu().numpy()
	label = np.asarray(label)
	label = label.copy()
	label[label >= (0.5 + 0)] = 1
	label[label < (0.5 + 0)] = 0
	# print(label)
	match = np.all(label == attack_class) + 0
	sample = sample[0]
	# Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
	# sample += RESNET_MEAN
	sample=sample*255
	# sample = sample[..., ::-1].astype(np.uint8)
	# Convert array to image and save
	image = sample.permute(1,2,0)
	# print(image.shape)
	image = image.cpu().numpy().astype(np.uint8)
	image = Image.fromarray(image)
	no_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
	if isbest:
		str="best"
	else:
		str=""
	if not os.path.exists(os.path.join("images", tar_iamge_name[0:-4])):
		os.makedirs(os.path.join("images", tar_iamge_name[0:-4]))

	# Save with predicted label for image (may not be adversarial due to uint8 conversion)
	l2 = round(l2, 3)
	image.save(os.path.join("images", tar_iamge_name[0:-4], "{}_{:.3f}_{}_{}.png".format(match,l2,no_time,str)))



def preprocess(sample_path):
	"""Load and preprocess image file."""
	img = image.load_img(sample_path, target_size=(448, 448))
	x = image.img_to_array(img)
	# print(x)
	x = np.expand_dims(x, axis=0)
	# x = preprocess_input(x)
	x= x/255
	x = np.copy(x)
	x = torch.from_numpy(x).float()
	x = x.permute(0, 3, 1, 2)
	x = x.cuda()
	# print(x.shape)
	return x


def get_diff(sample_1, sample_2):
	"""Channel-wise norm of difference between samples."""
	s1 = sample_1.permute(0, 2,3,1)
	s2 = sample_2.permute(0, 2,3,1)
	# print(np.linalg.norm(s1.cpu() - s2.cpu(), axis=(1, 2)).shape)
	# print(np.linalg.norm(s1.cpu() - s2.cpu(), axis=(1, 2)))

	return np.linalg.norm(s1.cpu() - s2.cpu(), axis=(1, 2))


def save_json(name,adversarial_sample, folder,target_sample):
	adv=adversarial_sample.cpu().numpy()
	tar = target_sample.cpu().numpy()
	print(adversarial_sample.shape)
	print(target_sample.shape)
	batch_r = (adv - tar)
	batch_r_255 = ((adv / 2 + 0.5) * 255) - ((tar / 2 + 0.5) * 255)
	norm = [np.linalg.norm(r.flatten()) for r in batch_r]
	# batch_l1_norm = [np.linalg.norm(r,ord=1) for r in batch_r]
	rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
	l1_norm=(np.sum(np.abs(adv - tar), axis=(1, 2, 3)))
	# l1_norm.extend(batch_l1_norm)
	max_r = np.max(np.abs(adv - tar), axis=(1, 2, 3))
	mean = np.mean(np.abs(adv - tar), axis=(1, 2, 3))
	ans_list={
		"name":name,
		"l1_norm":float(l1_norm[0]),
		"l2_norm":float(norm[0]),
		"infiy_norm":float(max_r[0]),
		"rmsd":float(rmsd[0]),
		"mean":float(mean[0])
	}
	print(ans_list)
	with open("./JSON/"+name[0:-4]+".json", "w") as f:
		json.dump(ans_list, f)


def boundary_attack():
	use_gpu = torch.cuda.is_available()
	# Load model, images and other parameters
	classifier = inceptionv3_attack(num_classes=20,save_model_path='../checkpoint/mlliw/voc2007/model_best.pth.tar')
	classifier.eval()
	if use_gpu:
		classifier = classifier.cuda()
	tar_iamge_name = "1o.jpg"
	initial_sample = preprocess('./images/original/1oa.png')
	target_sample = preprocess('./images/original/'+tar_iamge_name)

	folder = 'Mutil-Label'
	with torch.no_grad():
		attack_class = classifier(initial_sample)
		target_class = classifier(target_sample)

	attack_class=attack_class.cpu().numpy()
	attack_class = np.asarray(attack_class)
	attack_class = attack_class.copy()
	attack_class[attack_class >= (0.5 + 0)] = 1
	attack_class[attack_class < (0.5 + 0)] = 0

	target_class = target_class.cpu().numpy()
	target_class = np.asarray(target_class)
	target_class = target_class.copy()
	target_class[target_class >= (0.5 + 0)] = 1
	target_class[target_class < (0.5 + 0)] = 0

	# print(attack_class)
	# print(target_class)

	save_image(tar_iamge_name,initial_sample, classifier, folder,attack_class,0,0)

	adversarial_sample = initial_sample
	n_steps = 0
	n_calls = 0
	epsilon = 1.
	delta = 0.1
	bestl2=1000.

	# Move first step to the boundary 将第一步移动到边界
	while True:
		trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)
		with torch.no_grad():
			prediction = classifier(trial_sample)
		prediction = prediction.cpu().numpy()
		prediction = np.asarray(prediction)
		prediction = prediction.copy()
		prediction[prediction >= (0.5 + 0)] = 1
		prediction[prediction < (0.5 + 0)] = 0
		# print(prediction)
		n_calls += 1
		if np.all(prediction == attack_class):
			adversarial_sample = trial_sample
			print("初始化完毕")
			break
		else:
			epsilon *= 0.9

	# Iteratively run attack 迭代运行攻击
	while True:
		print("Step #{}...".format(n_steps))
		# Orthogonal step
		print("\tDelta step...")
		d_step = 0
		while True:
			d_step += 1
			print("\t#{}".format(d_step))
			trial_samples = []
			for i in np.arange(10):
				trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
				# trial_sample = trial_sample.squeeze()
				trial_samples.append(trial_sample)
			# trial_samples = np.array(trial_samples)
			d_score=0
			# print(trial_samples.shape)
			for i in np.arange(10):
				# print(trial_samples[i].shape)
				with torch.no_grad():
					prediction = classifier(trial_samples[i])
				prediction = prediction.cpu().numpy()
				prediction = np.asarray(prediction)
				prediction = prediction.copy()
				prediction[prediction >= (0.5 + 0)] = 1
				prediction[prediction < (0.5 + 0)] = 0
				d_score = d_score+(np.all(prediction == attack_class))/10
			n_calls += 10
			if d_score > 0.0:
				if d_score < 0.3:
					delta *= 0.9
				elif d_score > 0.7:
					delta /= 0.9
				# print("np.array(trial_samples)[].shape=",np.array(trial_samples).shape)
				for i in np.arange(10):
					with torch.no_grad():
						prediction = classifier(trial_samples[i])
					prediction = prediction.cpu().numpy()
					prediction = np.asarray(prediction)
					prediction = prediction.copy()
					prediction[prediction >= (0.5 + 0)] = 1
					prediction[prediction < (0.5 + 0)] = 0
					if(np.all(prediction == attack_class)) :
						adversarial_sample = trial_samples[i] #有问题
				break
			else:
				delta *= 0.9
		# Forward step
		print("\tEpsilon step...")
		e_step = 0
		while True:
			e_step += 1
			print("\t#{}".format(e_step))
			trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)
			with torch.no_grad():
				prediction = classifier(trial_sample)
			prediction = prediction.cpu().numpy()
			prediction = np.asarray(prediction)
			prediction = prediction.copy()
			prediction[prediction >= (0.5 + 0)] = 1
			prediction[prediction < (0.5 + 0)] = 0

			n_calls += 1
			if np.all(prediction == attack_class):
				adversarial_sample = trial_sample
				epsilon /= 0.5
				break
			elif e_step > 500:
					break
			else:
				epsilon *= 0.5

		n_steps += 1
		chkpts = [1, 5, 10, 50, 100, 500]
		diff = np.mean(get_diff(adversarial_sample, target_sample))
		if (n_steps in chkpts) or (n_steps % 500 == 0):
			print("{} steps".format(n_steps))
			save_image(tar_iamge_name,adversarial_sample, classifier, folder,attack_class,diff,0)
		if diff < bestl2:
			save_image(tar_iamge_name,adversarial_sample, classifier, folder, attack_class,diff,True)
			save_json(tar_iamge_name,adversarial_sample.cpu(), folder,target_sample.cpu())
			bestl2=diff
		if diff <= 1 or e_step > 500:
			print("{} steps".format(n_steps))
			print("Mean Squared Error: {}".format(diff))
			print("Calls: {}".format(n_calls))
			print("Attack Class: {}".format(attack_class))
			print("Target Class: {}".format(target_class))
			print("Adversarial Class: {}".format(np.argmax(prediction)))
			save_image(tar_iamge_name,adversarial_sample, classifier, folder,attack_class,diff,0)
			break

		print("Mean Squared Error: {}".format(diff))
		print("Calls: {}".format(n_calls))
		print("Attack Class: {}".format(attack_class))
		print("Target Class: {}".format(target_class))
		print("Adversarial Class: {}".format(np.argmax(prediction)))
		if(n_calls>=1000000):
			break;


if __name__ == "__main__":
	boundary_attack()
