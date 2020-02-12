2020.2.10 
Because I change the placement of direcctory, all codes below needs to be revised lightly

1. data_process.py
Step 1 : delete named dummy file
Step 2 : remove redundant depth information (we pre-filter rgb framews in local)
Step 3 : reorder (start from 00000)
Step 4 : save depth txt -> numpy

2. depth_inpaint.py
inpaint depth
	2.1 transform to greyscale image (.txt -> .npy -> .png)
	2.2 use cv2.inpaint to inpaint greyscale image
	2.3 visulaize to check (.png)
	2.4 transform back to actual depth info. (.npy)

3. mask_fill_zero.py
fill all black mask to 
    no corresponding frame -> mask
    no existence of directory of mask

4. data_augment.py
Do data augmentation on dataset_original 
Use 
    1. horizontally flip 
    2. change brightness 
    3. video clip of consecutive frame sequences

5. generate_train_test_list.py
Genrate train / test data list with 8 : 2 proportion
