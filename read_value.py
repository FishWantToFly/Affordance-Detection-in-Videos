import cv2, glob, os
import copy

# img = cv2.imread("00015.png")

x = 0
y = 0
y2 = 480
w = 640
h = 480

# img_name = "./dataset/chair/depth_frames/00328.png"
# img_name = "./inpaint.png"
# img = cv2.imread(img_name, 0)

# for i in range (img.shape[0]): #traverses through height of the image
#     for j in range (img.shape[1]): #traverses through width of the image
#         print (img[i][j])

# =======================================
# ## inpaint
theme = 'bottle'
rgb_frames_inpaint_dir = "./dataset/%s/depth_frames_inpaint" % (theme)
imgs_list = glob.glob("./dataset/%s/depth_frames/*.png" % (theme))
imgs_list = sorted(imgs_list)

if not os.path.exists(rgb_frames_inpaint_dir):
	os.mkdir(rgb_frames_inpaint_dir)

for img_name in imgs_list :
	img_name_pure = img_name.split("/")[-1]
	img = cv2.imread(img_name, 0)

	mask = copy.deepcopy(img)
	for i in range (mask.shape[0]): #traverses through height of the image
		for j in range (mask.shape[1]): #traverses through width of the image
			if mask[i][j] != 0:
				mask[i][j] = 0
			else :
				mask[i][j] = 255

	inpaint_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
	cv2.imwrite(os.path.join(rgb_frames_inpaint_dir, img_name_pure), inpaint_img)

# ========================================
# show value

# imgs_list = glob.glob("./dataset/test/depth_frames/*.png")
# imgs_list = sorted(imgs_list)

# for img_name in imgs_list :
# 	img = cv2.imread(img_name, 0)
# 	print("%s %d" % (img_name, img[275][390]))

# 390 275