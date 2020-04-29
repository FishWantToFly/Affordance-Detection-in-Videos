'''
2_deal_with_mask.py
1. copy mask -> first_mask & mask_rgb -> first_mask_rgb
    mask :
        if no first_mask and first_mask_rgb : (all have true affordance / all have no affordance but hard to label)
            copy mask from dataset
        if have first_mask and first_mask_rgb :
            check data number is same with dataset
                if no -> copy mask and mask_rgb from dataset
        in the end, we can get new mask for semantic segmentation
    mask_rgb : 
        same as mask

'''

import glob, os

dataset_source = '../dataset/dataset_original'
mask_name = 'mask'
mask_rgb_name = 'mask_rgb'
first_mask_name = 'first_mask'
first_mask_rgb_name = 'first_mask_rgb'

print("Step 1")
actions = sorted(glob.glob("./dataset_original/*/*/*"))
for action in actions:
    _object = action.split('/')[3].split('_')[0]
    target_list = ['chair', 'table']
    if _object not in target_list:
        continue
        
    # source_action_path = os.path.join(dataset_source, ('/').join(action.split('/')[2:5]))
    
    # for test
    # action = './dataset_original/home_living_room/basket/put_object_on_it_1'
    print(action)

    mask_path =  os.path.join(action, mask_name) # old
    mask_rgb_path = os.path.join(action, mask_rgb_name) # old
    first_mask_path = os.path.join(action, first_mask_name)
    first_mask_rgb_path = os.path.join(action, first_mask_rgb_name)



    # 1. chehck whether there are first_mask and first_mask_rgb
    if not os.path.exists(os.path.join(action, first_mask_name)) and not os.path.exists(os.path.join(action, first_mask_rgb_name)) :
        # directly copy from dataset_source
        os.system("cp -r %s %s" % (mask_path, first_mask_path))
        os.system("cp -r %s %s" % (mask_rgb_path, first_mask_rgb_path))
    else :
        # 2. check if there is missing first_mask_rgb. If miss, copy from mask_rgb
        # 3. check if there is missing first_mask. If miss, copy from mask

        # List raw_frames and map to mask and mask_rgb
        frames = sorted(glob.glob(os.path.join(action, 'raw_frames/*')))
        for frame in frames : 
            frame_name = os.path.split(frame)[1]
            first_mask_now = os.path.join(action, first_mask_name, frame_name[:-4] + '.jpg')
            first_mask_rgb_now = os.path.join(action, first_mask_rgb_name, frame_name[:-4] + '.jpg')
            mask_now = os.path.join(action, mask_name, frame_name[:-4] + '.jpg')
            mask_rgb_now = os.path.join(action, mask_rgb_name, frame_name[:-4] + '.jpg')

            # chech first_mask_rgb
            if not os.path.exists(first_mask_rgb_now):
                os.system("cp -r %s %s" % (mask_rgb_now, first_mask_rgb_now))
            # chech first_mask
            if not os.path.exists(first_mask_now):
                os.system("cp -r %s %s" % (mask_now, first_mask_now))

