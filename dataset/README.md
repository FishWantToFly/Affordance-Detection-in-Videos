2020.2.10 

Because I change the placement of direcctory, all codes below needs to be revised lightly.
---
* demo.py (draw mask. not in this directory now)
    * Detect instance single by single
    1. pass


* 1_data_process.py
    1. delete named dummy file
    2. remove redundant depth information (we pre-filter rgb framews in local)
    3. reorder (start from 00000)
    4. save depth txt -> numpy

* 2_depth_inpaint.py
    1. transform to greyscale image (.txt -> .npy -> .png)
    2. use cv2.inpaint to inpaint greyscale image
    3. visulaize to check (.png)
    4. transform back to actual depth info. (.npy)

* 3_mask_fill_zero.py
    fill all black mask to 
    1. no corresponding frame -> mask
    2. no existence of directory of mask

* 4_data_augment.py
    Do data augmentation on dataset_original 
    Use 
        1. horizontally flip 
        2. change brightness 
        3. video clip of consecutive frame sequences

* 5_generate_train_test_list.py
    Genrate train / test data list with 8 : 2 proportion
