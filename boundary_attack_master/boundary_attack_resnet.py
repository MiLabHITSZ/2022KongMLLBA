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

	# Project perturbation onto sphere around target
	diff = (target_sample.permute(0, 2,3,1) - prev_sample.permute(0, 2,3,1)).cpu() # Orthorgonal vector to sphere surface
	diff /= get_diff(target_sample, prev_sample) # Orthogonal unit vector
	# We project onto the orthogonal then subtract from perturb
	# to get projection onto sphere surface
	perturb = torch.from_numpy(perturb).float()
	perturb= perturb.permute(0, 3, 1, 2)
	diff = diff.permute(0, 3, 1, 2)
	perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff)**2) * diff
	# Check overflow and underflow
	perturb = np.clip(perturb, 0., 1.)
	return perturb.cuda()


def forward_perturbation(epsilon, prev_sample, target_sample):
	"""Generate forward perturbation."""
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
	sample=torch.from_numpy(sample).float()
	sample= sample.permute(0, 3, 1, 2)
	sample=sample.cuda()
	with torch.no_grad():
		label = classifier(sample)
	return label


def save_image(image_index, adversarial_sample, l2,save_image_folder):
	"""Export image file."""
	sample = adversarial_sample[0]
	# Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
	# sample += RESNET_MEAN
	sample=sample*255
	# sample = sample[..., ::-1].astype(np.uint8)
	# Convert array to image and save
	image = sample.permute(1,2,0)
	image = image.cpu().numpy().astype(np.uint8)
	image = Image.fromarray(image)
	image.save(os.path.join("../boundary_attack_master/images", save_image_folder, "{}_fin.png".format(image_index,str)))



def preprocess(sample_path):
	"""Load and preprocess image file."""
	img = image.load_img(sample_path, target_size=(448, 448))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x= x/255
	x = np.copy(x)
	x = torch.from_numpy(x).float()
	x = x.permute(0, 3, 1, 2)
	x = x.cuda()
	return x


def get_diff(sample_1, sample_2):
	"""Channel-wise norm of difference between samples."""
	s1 = sample_1.permute(0, 2,3,1)
	s2 = sample_2.permute(0, 2,3,1)

	return np.linalg.norm(s1.cpu() - s2.cpu(), axis=(1, 2))


def save_json(name,adversarial_sample, save_image_folder,target_sample,best_l2):
	adv=adversarial_sample.cpu().numpy()
	tar = target_sample.cpu().numpy()
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
	print("A better adversarial example is generated.")
	print(ans_list)
	if(norm[0] < best_l2):
		with open("../boundary_attack_master/JSON/"+save_image_folder+"/"+str(name)+".json", "w") as f:
			json.dump(ans_list, f)
	return  norm[0]


def boundary_attack(initial_sample,target_sample,model,max_test,l2_threshold,image_index,save_image_folder,best_l2):
	use_gpu = torch.cuda.is_available()
	# Load model, images and other parameters
	classifier = model
	classifier.eval()
	if use_gpu:
		classifier = classifier.cuda()

	initial_sample = preprocess(initial_sample)
	target_sample = preprocess(target_sample)

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

	adversarial_sample = initial_sample
	n_steps = 0
	n_calls = 0
	epsilon = 1.
	delta = 0.1
	best_tem=1000.

	# Move first step to the boundary
	while True:
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
			break
		else:
			epsilon *= 0.9

	# Iteratively run attack
	while True:
		# Orthogonal step
		d_step = 0
		while True:
			d_step += 1
			trial_samples = []
			for i in np.arange(10):
				trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
				# trial_sample = trial_sample.squeeze()
				trial_samples.append(trial_sample)
			# trial_samples = np.array(trial_samples)
			d_score=0
			for i in np.arange(10):
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
				for i in np.arange(10):
					with torch.no_grad():
						prediction = classifier(trial_samples[i])
					prediction = prediction.cpu().numpy()
					prediction = np.asarray(prediction)
					prediction = prediction.copy()
					prediction[prediction >= (0.5 + 0)] = 1
					prediction[prediction < (0.5 + 0)] = 0
					if(np.all(prediction == attack_class)) :
						adversarial_sample = trial_samples[i]
				break
			else:
				delta *= 0.9
		# Forward step
		e_step = 0
		while True:
			e_step += 1
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
		with torch.no_grad():
			label = classifier(adversarial_sample)
		label = label.cpu().numpy()
		label = np.asarray(label)
		label = label.copy()
		label[label >= (0.5 + 0)] = 1
		label[label < (0.5 + 0)] = 0
		match = np.all(label == attack_class) + 0
		if(match==1):
			diff = np.mean(get_diff(adversarial_sample, target_sample))
		else:
			return 1000,n_calls
		if diff < best_tem:
			# save_image(tar_iamge_name,adversarial_sample, classifier, folder, attack_class,diff,True)
			l2 = save_json(image_index,adversarial_sample.cpu(), save_image_folder,target_sample.cpu(),best_l2)
			save_image(image_index, adversarial_sample, l2, save_image_folder)
			best_tem=diff
		if(l2<=1 or n_calls >= max_test or l2>= 150 or diff>=150):
			break;
	return l2,n_calls


if __name__ == "__main__":
	boundary_attack()
