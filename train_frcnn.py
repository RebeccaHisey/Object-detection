from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from tensorflow.keras import utils

sys.setrecursionlimit(40000)

parser = OptionParser()

#parser.add_option("-p", "--path", dest="train_path", help="Path to training data.",default='D:/Pilot_Study/MS01-20200210-135709_adapted.txt')
parser.add_option("-t","--train_videos", dest="train_videos", help="Path to text file containing video names to be used for training", default = 'annotation_files/train.txt')
parser.add_option("--val", '--validation_videos', dest="validation_videos", default = 'annotation_files/vaildation.txt')
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="simple")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()
'''
if not options.train_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')
'''
if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

if options.network == 'vgg':
	C.network = 'vgg'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
else:
	print('Not a valid model')
	raise ValueError


# check if weight path was passed via command line
if options.input_weight_path:
	C.base_net_weights = options.input_weight_path
else:
	# set the path to weights based on backend and model
	C.base_net_weights = nn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(options.train_videos, options.validation_videos, True)
all_val_imgs, val_classes_count, val_class_mapping = get_data(options.train_videos, options.validation_videos, False)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

if 'bg' not in val_classes_count:
	val_classes_count['bg'] = 0
	val_class_mapping['bg'] = len(val_class_mapping)

C.class_mapping = class_mapping
C.val_class_mapping = val_class_mapping

inv_map = {v: k for k, v in class_mapping.items()}
val_inv_map = {v: k for k, v in val_class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

print('validation images per class:')
pprint.pprint(val_classes_count)
print('Num val classes (including bg) = {}'.format(len(val_classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)
random.shuffle(all_val_imgs)

num_imgs = len(all_imgs)
num_val_imgs = len(all_val_imgs)

train_imgs = [s for s in all_imgs]
val_imgs = [s for s in all_val_imgs]

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, mode='val')


input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
	print('loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = len(all_imgs)
val_epoch_length = len(all_val_imgs)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
val_losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor_val = []
rpn_accuracy_for_epoch_val = []
start_time = time.time()

best_loss = np.Inf
best_acc = -1

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True
val_loss_decreasing = True
val_acc_increasing = True
epoch_num = 0

while (val_loss_decreasing and val_acc_increasing and epoch_num < num_epochs) or (epoch_num < 20):
#for epoch_num in range(num_epochs):
	epoch_num +=1
	progbar = utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num, num_epochs))

	while True:
		try:

			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			X, Y, img_data = next(data_gen_train)
			#Xval, Yval, img_data_val = next(data_gen_val)

			loss_rpn = model_rpn.train_on_batch(X, Y)

			P_rpn = model_rpn.predict_on_batch(X)

			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)
			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []
			
			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			if C.num_rois > 1:
				if len(pos_samples) < C.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
				try:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples
			else:
				# in the extreme case where num_rois = 1, we pick a random pos or neg sample
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = random.choice(neg_samples)
				else:
					sel_samples = random.choice(pos_samples)

			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
									  ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

			iter_num += 1
			
			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
						mean_overlapping_bboxes))
					print('Train Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Train Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Train Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Train Loss Detector classifier: {}'.format(loss_class_cls))
					print('Train Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))
				iter_num = 0
				val_iternum = 0
				while val_iternum < val_epoch_length:
					Xval, Yval, img_data_val = next(data_gen_val)
					loss_rpn_val = model_rpn.test_on_batch(Xval, Yval)
					P_rpn_val = model_rpn.predict_on_batch(Xval)

					Rval = roi_helpers.rpn_to_roi(P_rpn_val[0], P_rpn_val[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)
					# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
					X2val, Y1val, Y2val, IouSval = roi_helpers.calc_iou(Rval, img_data_val, C, class_mapping)

					if X2val is None:
						rpn_accuracy_rpn_monitor_val.append(0)
						rpn_accuracy_for_epoch_val.append(0)
						continue

					neg_samples = np.where(Y1val[0, :, -1] == 1)
					pos_samples = np.where(Y1val[0, :, -1] == 0)

					if len(neg_samples) > 0:
						neg_samples = neg_samples[0]
					else:
						neg_samples = []

					if len(pos_samples) > 0:
						pos_samples = pos_samples[0]
					else:
						pos_samples = []

					rpn_accuracy_rpn_monitor_val.append(len(pos_samples))
					rpn_accuracy_for_epoch_val.append((len(pos_samples)))

					if C.num_rois > 1:
						if len(pos_samples) < C.num_rois // 2:
							selected_pos_samples = pos_samples.tolist()
						else:
							selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
						try:
							selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
																	replace=False).tolist()
						except:
							selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
																	replace=True).tolist()

						sel_samples = selected_pos_samples + selected_neg_samples
					else:
						# in the extreme case where num_rois = 1, we pick a random pos or neg sample
						selected_pos_samples = pos_samples.tolist()
						selected_neg_samples = neg_samples.tolist()
						if np.random.randint(0, 2):
							sel_samples = random.choice(neg_samples)
						else:
							sel_samples = random.choice(pos_samples)

					loss_class_val = model_classifier.test_on_batch([Xval, X2val[:, sel_samples, :]],
																 [Y1val[:, sel_samples, :], Y2val[:, sel_samples, :]])

					val_losses[val_iternum, 0] = loss_rpn_val[1]
					val_losses[val_iternum, 1] = loss_rpn_val[2]

					val_losses[val_iternum, 2] = loss_class_val[1]
					val_losses[val_iternum, 3] = loss_class_val[2]
					val_losses[val_iternum, 4] = loss_class_val[3]

					val_iternum+=1

				loss_rpn_cls = np.mean(val_losses[:, 0])
				loss_rpn_regr = np.mean(val_losses[:, 1])
				loss_class_cls = np.mean(val_losses[:, 2])
				loss_class_regr = np.mean(val_losses[:, 3])
				class_acc = np.mean(val_losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch_val)) / len(rpn_accuracy_for_epoch_val)
				rpn_accuracy_for_epoch_val = []

				if C.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Validation Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Validation Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Validation Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Validation Loss Detector classifier: {}'.format(loss_class_cls))
					print('Validation Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					if C.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(C.model_path)
					if class_acc > best_acc:
						best_acc = class_acc
				elif class_acc > best_acc:
					if C.verbose:
						print('Total accuracy increased from {} to {}, saving weights'.format(best_acc, class_acc))
					best_acc = class_acc
					model_all.save_weights(C.model_path)
				else:
					val_acc_increasing = False
					val_loss_decreasing = False
				break

		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete, exiting.')
