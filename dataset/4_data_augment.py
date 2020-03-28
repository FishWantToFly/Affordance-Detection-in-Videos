'''
Do data augmentation on dataset_original 
Use 
    1. horizontally flip 
		Do on dataset_original only
		Need to copy : 
			mask -> flip
			inpaint_depth -> flip
			raw_frames -> flip
    2. change brightness 
    3. video clip of consecutive frame sequences
'''

import glob, os, copy, random
import numpy as np
from os import walk
from PIL import Image, ImageOps, ImageEnhance

def create_dir(new_dir):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)


depth_dir = 'raw_depth'
frame_dir = 'raw_frames'
mask_dir = 'mask'
depth_dir = 'inpaint_depth'
now_dir = os.getcwd()
IMG_HEIGHT = 480
IMG_WIDTH = 640


# ######################################################
# # Step 1 : horizontally flip 
# # test_action = "/home/s5078345/Affordance-Detection-on-Video/dataset/dataset_original/lab/chair_1/put_big_object_on_it"
# copy_dir_name = 'dataset_horizontally_flip'
# if not os.path.exists(copy_dir_name):
# 	os.mkdir(copy_dir_name)

# for action in glob.glob("./dataset_original/*/*/*"):
# 	_, _action = os.path.split(action)
# 	print(action)

# 	# 1. create directory
# 	place = action.split('/')[-3]
# 	_object = action.split('/')[-2]
# 	# print(place)
# 	# print(_object)
# 	new_place_dir = os.path.join(now_dir, copy_dir_name, place)
# 	new_object_dir = os.path.join(now_dir, copy_dir_name, place, _object)
# 	if not os.path.exists(new_place_dir):
# 		os.mkdir(new_place_dir)
# 	if not os.path.exists(new_object_dir):
# 		os.mkdir(new_object_dir)

# 	action_mask_path = os.path.join(action, mask_dir, '*')
# 	action_frame_path = os.path.join(action, frame_dir, '*')
# 	action_depth_path = os.path.join(action, depth_dir, '*')
# 	mask_list = sorted(glob.glob(action_mask_path))
# 	frame_list = sorted(glob.glob(action_frame_path))
# 	depth_list = sorted(glob.glob(action_depth_path))

# 	new_action_path = os.path.join(now_dir, copy_dir_name, place, _object, _action)
# 	create_dir(new_action_path)
# 	new_mask_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, mask_dir)
# 	new_depth_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, depth_dir)
# 	new_frame_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, frame_dir)
# 	create_dir(new_mask_path)
# 	create_dir(new_depth_path)
# 	create_dir(new_frame_path)

# 	# 2. flip mask and frame, and then copy
# 	for mask in mask_list:
# 		_, _mask = os.path.split(mask)
# 		mask_img = Image.open(mask)
# 		img_mirror = ImageOps.mirror(mask_img)
# 		img_mirror_path = os.path.join(new_mask_path, _mask)
# 		img_mirror.save(img_mirror_path, quality=95)

# 	for frame in frame_list:
# 		_, _frame = os.path.split(frame)
# 		frame_img = Image.open(frame)
# 		frame_mirror = ImageOps.mirror(frame_img)
# 		frame_mirror_path = os.path.join(new_frame_path, _frame)
# 		frame_mirror.save(frame_mirror_path, quality=95)

# 	# 3. flip depth then copy
# 	for depth in depth_list:
# 		_, _depth = os.path.split(depth)
# 		depth_npy = np.load(depth)
# 		depth_npy = depth_npy.reshape(480, 640)
# 		depth_mirror = np.fliplr(depth_npy)
# 		depth_mirror = depth_mirror.reshape(480, 640, 1)
# 		depth_mirror_path = os.path.join(new_depth_path, _depth)
# 		np.save(depth_mirror_path, depth_mirror)

# ######################################################
# # Step 2 : adjust brightness or contrast
# # test_action = "/home/s5078345/affordance/dataset/dataset_original/lab/chair_1/put_big_object_on_it"
# os.remove("error.txt")
# fo = open("error.txt", "w")
# fo.write("Failuse case for 4_data_augment.py")

# # 1.
# # source_dir_name = 'dataset_original'
# # copy_dir_name = 'dataset_brightness_contrast'
# # 2.
# source_dir_name = 'dataset_horizontally_flip'
# copy_dir_name = 'dataset_flip_brightness_contrast'

# if not os.path.exists(copy_dir_name):
# 	os.mkdir(copy_dir_name)

# for action in sorted(glob.glob("./%s/*/*/*" % (source_dir_name))):
# 	# action = test_action
# 	_, _action = os.path.split(action)
# 	print(action)

# 	# 1. create directory
# 	place = action.split('/')[-3]
# 	_object = action.split('/')[-2]
# 	# print(place)
# 	# print(_object)
# 	new_place_dir = os.path.join(now_dir, copy_dir_name, place)
# 	new_object_dir = os.path.join(now_dir, copy_dir_name, place, _object)
# 	if not os.path.exists(new_place_dir):
# 		os.mkdir(new_place_dir)
# 	if not os.path.exists(new_object_dir):
# 		os.mkdir(new_object_dir)

# 	action_mask_path = os.path.join(action, mask_dir, '*')
# 	action_frame_path = os.path.join(action, frame_dir, '*')
# 	action_depth_path = os.path.join(action, depth_dir, '*')
# 	mask_list = sorted(glob.glob(action_mask_path))
# 	frame_list = sorted(glob.glob(action_frame_path))
# 	depth_list = sorted(glob.glob(action_depth_path))

# 	new_action_path = os.path.join(now_dir, copy_dir_name, place, _object, _action)
# 	create_dir(new_action_path)
# 	new_mask_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, mask_dir)
# 	new_depth_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, depth_dir)
# 	new_frame_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, frame_dir)
# 	create_dir(new_mask_path)
# 	create_dir(new_depth_path)
# 	create_dir(new_frame_path)

# 	random.seed()
# 	random_1 = random.random()
# 	random_2 = random.random()

# 	for frame in frame_list:
# 		_, _frame = os.path.split(frame)
# 		frame_img = Image.open(frame)

# 		try:
# 			# Adjust sharpness or brightnesss
# 			if random_1 > 0.6 :
# 				frame_img = ImageEnhance.Sharpness(frame_img)
# 				frame_img = frame_img.enhance(10.0)		
# 			else :
# 				frame_img = ImageEnhance.Brightness(frame_img)
# 				if random_2 > 0.5 :
# 					frame_img = frame_img.enhance(1.35)
# 				else :
# 					frame_img = frame_img.enhance(0.8)
# 		except:
# 			fo.write(action)
# 			break

		
# 		# save adjusted img
# 		enhance_frame_path = os.path.join(new_frame_path, _frame)
# 		frame_img.save(enhance_frame_path, quality=95)

# 	# 2. just copy mask (no change)
# 	for mask in mask_list:
# 		_, _mask = os.path.split(mask)
# 		mask_img = Image.open(mask)
# 		img_mirror = mask_img
# 		img_mirror_path = os.path.join(new_mask_path, _mask)
# 		img_mirror.save(img_mirror_path, quality=95)

# 	# 3. just copy depth (no change)
# 	for depth in depth_list:
# 		_, _depth = os.path.split(depth)
# 		depth_npy = np.load(depth)
# 		depth_npy = depth_npy.reshape(480, 640)
# 		depth_mirror = depth_npy
# 		depth_mirror = depth_mirror.reshape(480, 640, 1)
# 		depth_mirror_path = os.path.join(new_depth_path, _depth)
# 		np.save(depth_mirror_path, depth_mirror)
# fo.close()

######################################################
# Step 3 : random shift + occlusion (_so)
test_action = "/home/s5078345/Affordance-Detection-on-Video/dataset/dataset_original/lab/chair_1/put_big_object_on_it"
datasets = ['dataset_original', 'dataset_horizontally_flip', 'dataset_flip_brightness_contrast', 'dataset_brightness_contrast']

idx = -1

for dataset in datasets:
	copy_dir_name = dataset + '_so'
	if not os.path.exists(copy_dir_name):
		os.mkdir(copy_dir_name)

	for action in sorted(glob.glob("./%s/*/*/*" % (dataset))):
		idx += 1

		# action = test_action
		_, _action = os.path.split(action)
		print(action)

		# 1. create directory
		place = action.split('/')[-3]
		_object = action.split('/')[-2]
		# print(place)
		# print(_object)
		new_place_dir = os.path.join(now_dir, copy_dir_name, place)
		new_object_dir = os.path.join(now_dir, copy_dir_name, place, _object)
		if not os.path.exists(new_place_dir):
			os.mkdir(new_place_dir)
		if not os.path.exists(new_object_dir):
			os.mkdir(new_object_dir)

		action_mask_path = os.path.join(action, mask_dir, '*')
		action_frame_path = os.path.join(action, frame_dir, '*')
		action_depth_path = os.path.join(action, depth_dir, '*')
		mask_list = sorted(glob.glob(action_mask_path))
		frame_list = sorted(glob.glob(action_frame_path))
		depth_list = sorted(glob.glob(action_depth_path))

		new_action_path = os.path.join(now_dir, copy_dir_name, place, _object, _action)
		create_dir(new_action_path)
		new_mask_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, mask_dir)
		new_depth_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, depth_dir)
		new_frame_path = os.path.join(now_dir, copy_dir_name, place, _object, _action, frame_dir)
		create_dir(new_mask_path)
		create_dir(new_depth_path)
		create_dir(new_frame_path)

		random.seed()
		random_1 = random.random()
		# random_1 = 0.2

		# for random shift
		h_num = int (480 / 6)
		height_shift = random.randint(-h_num, h_num)
		w_num = int (640 / 6)
		width_shift = random.randint(-w_num, w_num)

		for frame in frame_list:
			_frame_dir, _frame = os.path.split(frame)
			frame_img = Image.open(frame)
			np_img = np.asarray(frame_img, dtype='uint8')

			# Read files

			# mask
			old_mask_path = os.path.join(os.path.dirname(_frame_dir), mask_dir, _frame[:-4] + '.jpg')
			mask_img = Image.open(old_mask_path)
			mask_np = np.asarray(mask_img, dtype='uint8')
			if mask_np.ndim == 2 :
				mask_np = np.expand_dims(mask_np, axis=-1)

			# depth
			old_depth_path = os.path.join(os.path.dirname(_frame_dir), depth_dir, _frame[:-4] + '.npy')
			depth_npy = np.load(old_depth_path)
			depth_npy = depth_npy.reshape(480, 640)
			# np.save(new_depth_path, depth_mirror)

			# random shift
			if random_1 > 0.5 :
				# initialize
				shift_np_img = np.full_like(np_img, 0) # 480 640 3
				shift_mask = np.full_like(mask_np, 0)
				shift_depth = np.full_like(depth_npy, 0)

				if height_shift >= 0 and width_shift >= 0:
					shift_np_img[height_shift:IMG_HEIGHT, width_shift:IMG_WIDTH, :] = np_img[0: (IMG_HEIGHT - height_shift), \
						0: (IMG_WIDTH - width_shift), :]
					shift_mask[height_shift:IMG_HEIGHT, width_shift:IMG_WIDTH, :] = mask_np[0: (IMG_HEIGHT - height_shift), \
						0: (IMG_WIDTH - width_shift), :]
					shift_depth[height_shift:IMG_HEIGHT, width_shift:IMG_WIDTH] = depth_npy[0: (IMG_HEIGHT - height_shift), \
						0: (IMG_WIDTH - width_shift)]
				elif height_shift < 0 and width_shift >= 0:
					shift_np_img[0:(IMG_HEIGHT + height_shift), width_shift:IMG_WIDTH, :] = np_img[-height_shift: IMG_HEIGHT, \
						0: (IMG_WIDTH - width_shift), :]
					shift_mask[0:(IMG_HEIGHT + height_shift), width_shift:IMG_WIDTH, :] = mask_np[-height_shift: IMG_HEIGHT, \
						0: (IMG_WIDTH - width_shift), :]
					shift_depth[0:(IMG_HEIGHT + height_shift), width_shift:IMG_WIDTH] = depth_npy[-height_shift: IMG_HEIGHT, \
						0: (IMG_WIDTH - width_shift)]
				elif height_shift >= 0 and width_shift < 0:
					shift_np_img[height_shift:IMG_HEIGHT, 0:(IMG_WIDTH + width_shift), :] = np_img[0: (IMG_HEIGHT - height_shift), \
						-width_shift: IMG_WIDTH, :]
					shift_mask[height_shift:IMG_HEIGHT, 0:(IMG_WIDTH + width_shift), :] = mask_np[0: (IMG_HEIGHT - height_shift), \
						-width_shift: IMG_WIDTH, :]
					shift_depth[height_shift:IMG_HEIGHT, 0:(IMG_WIDTH + width_shift)] = depth_npy[0: (IMG_HEIGHT - height_shift), \
						-width_shift: IMG_WIDTH]
				elif height_shift < 0 and width_shift < 0:
					shift_np_img[0:(IMG_HEIGHT + height_shift), 0:(IMG_WIDTH + width_shift), :] = np_img[0: (IMG_HEIGHT + height_shift), \
						0: (IMG_WIDTH + width_shift), :]
					shift_mask[0:(IMG_HEIGHT + height_shift), 0:(IMG_WIDTH + width_shift), :] = mask_np[0: (IMG_HEIGHT + height_shift), \
						0: (IMG_WIDTH + width_shift), :]
					shift_depth[0:(IMG_HEIGHT + height_shift), 0:(IMG_WIDTH + width_shift)] = depth_npy[0: (IMG_HEIGHT + height_shift), \
						0: (IMG_WIDTH + width_shift)]

				shift_img = Image.fromarray(np.uint8(shift_np_img))
				shift_img_path = os.path.join(new_frame_path, _frame)
				shift_img.save(shift_img_path, quality=95)

				# mask
				if shift_mask.shape[2] == 1 :
					shift_mask = np.squeeze(shift_mask, axis=-1)
				shift_mask = Image.fromarray(np.uint8(shift_mask))
				shift_mask.save(os.path.join(new_mask_path, _frame[:-4] + '.jpg'), quality=95)

				# depth
				shift_depth = shift_depth.reshape(480, 640, 1)
				np.save(os.path.join(new_depth_path, _frame[:-4] + '.npy'), shift_depth)

			else : 
				# occupancy
				height_shift = random.randint(40, 120)
				width_shift = random.randint(40, 120)

				height_start = random.randint(40, 480 - 120)
				width_start = random.randint(40, 640 - 120)
				
				occupancy_np_img = np_img.copy()
				occupancy_mask = mask_np.copy()
				occupancy_depth = depth_npy.copy()

				occupancy_np_img[height_start: (height_start + height_shift), width_start: (width_start + width_shift), :] = 0
				if occupancy_mask.ndim == 2 :
					occupancy_mask[height_start: (height_start + height_shift), width_start: (width_start + width_shift)] = 0
				elif occupancy_mask.ndim == 3 :
					occupancy_mask[height_start: (height_start + height_shift), width_start: (width_start + width_shift), :] = 0
				occupancy_depth[height_start: (height_start + height_shift), width_start: (width_start + width_shift)] = 0

				occupancy_img = Image.fromarray(np.uint8(occupancy_np_img))
				occupancy_img_path = os.path.join(new_frame_path, _frame)
				occupancy_img.save(occupancy_img_path, quality=95)

				# mask
				if occupancy_mask.shape[2] == 1 :
					occupancy_mask = np.squeeze(occupancy_mask, axis=-1)
				occupancy_mask = Image.fromarray(np.uint8(occupancy_mask))
				occupancy_mask.save(os.path.join(new_mask_path, _frame[:-4] + '.jpg'), quality=95)

				# depth
				occupancy_depth = occupancy_depth.reshape(480, 640, 1)
				np.save(os.path.join(new_depth_path, _frame[:-4] + '.npy'), occupancy_depth)

