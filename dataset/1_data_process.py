import glob, os
from os import walk


# dataset_name = 'dataset_original'
dataset_name = 'dataset_lab_ito'

# # Step 1 : delete named dummy file
# f = []	
# for filename in glob.glob("./%s/*/*/*/dummy" % (dataset_name)):
# 	print(filename)
# 	os.remove(filename)

#######################################################
# # Step 2 : remove redundant depth information
# # test_action_dir = "/home/s5078345/affordance/dataset/kitchen/chair_1/remove_big_object_on_it_1/"
# depth_dir = 'raw_depth'
# frame_dir = 'raw_frames'

# # delete redundant depth
# for action in glob.glob("./%s/*/*/*" % (dataset_name)):
# 	print(action)
# 	action_depth_path = os.path.join(action, depth_dir, '*')
# 	action_frame_path = os.path.join(action, frame_dir, '*')
# 	depth_list = sorted(glob.glob(action_depth_path))
# 	frame_list = sorted(glob.glob(action_frame_path))

# 	for depth_txt in depth_list :
# 		# depth_txt_dir = ("/").join(depth_txt.split('/')[:-2])
# 		depth_txt_name = depth_txt.split('/')[-1]

# 		maybe_exist_frame_path = os.path.join(action, frame_dir, depth_txt_name[:-4] + '.png')
# 		# print(maybe_exist_frame_path)

# 		if not os.path.isfile(maybe_exist_frame_path):
# 			# print(maybe_exist_frame_path)
# 			# delete depth txt itself
# 			os.remove(depth_txt) 

#######################################################
# # Step 3 : reorder (start from 00000)
# # test_action_dir = "/home/s5078345/affordance/dataset/kitchen/chair_1/remove_big_object_on_it_1/"
# depth_dir = 'raw_depth'
# frame_dir = 'raw_frames'

# for action in glob.glob("./%s/*/*/*" % (dataset_name)):
# 	action_depth_path = os.path.join(action, depth_dir, '*')
# 	action_frame_path = os.path.join(action, frame_dir, '*')
# 	depth_list = sorted(glob.glob(action_depth_path))
# 	frame_list = sorted(glob.glob(action_frame_path))

# 	count = 0
# 	for depth_txt in depth_list :
# 		depth_txt_name = depth_txt.split('/')[-1] 
# 		frame_path = os.path.join(action, frame_dir, depth_txt_name[:-4] + '.png')

# 		new_depth_name =  os.path.join(action, depth_dir, "%05d" % (count) + '.txt') 
# 		new_frame_name = os.path.join(action, frame_dir, "%05d" % (count) + '.png') 
# 		count += 1

# 		os.rename(frame_path, new_frame_name)
# 		os.rename(depth_txt, new_depth_name)


######################################################
# Step 4 : save depth txt -> numpy
# test_action_dir = "/home/s5078345/affordance/dataset/kitchen/chair_1/remove_big_object_on_it_1/"
depth_dir = 'raw_depth'
frame_dir = 'raw_frames'

# delete redundant depth
for action in glob.glob("./%s/*/*/*" % (dataset_name)):
	print(action)
	action_depth_path = os.path.join(action, depth_dir, '*')
	action_frame_path = os.path.join(action, frame_dir, '*')
	depth_list = sorted(glob.glob(action_depth_path))
	frame_list = sorted(glob.glob(action_frame_path))

	count = 0
	for depth_txt in depth_list :
		depth_txt_name = depth_txt.split('/')[-1] 
		frame_path = os.path.join(action, frame_dir, depth_txt_name[:-4] + '.png')

		new_depth_name =  os.path.join(action, depth_dir, "%05d" % (count) + '.txt') 
		new_frame_name = os.path.join(action, frame_dir, "%05d" % (count) + '.png') 
		count += 1

		os.rename(frame_path, new_frame_name)
		os.rename(depth_txt, new_depth_name)