import cv2, glob, os

# img = cv2.imread("00015.png")

x = 0
y = 0
y2 = 480
w = 640
h = 480

# crop_img = img[y:y+h, x:x+w]
# crop_img2 = img[y2:y2+h, x:x+w]


# cv2.imwrite('crop.png', crop_img)
# cv2.imwrite('crop_2.png', crop_img2)

theme = 'bottle'

imgs_list = glob.glob("./dataset/%s/raw_frames/*.png" % (theme))
imgs_list = sorted(imgs_list)

rgb_frames_dir = "./dataset/%s/rgb_frames" % (theme)
depth_frames_dir = "./dataset/%s/depth_frames" % (theme)

if not os.path.exists(rgb_frames_dir):
	os.mkdir(rgb_frames_dir)
if not os.path.exists(depth_frames_dir):
	os.mkdir(depth_frames_dir)
# print(imgs_list)

for img_name in imgs_list :
	img = cv2.imread(img_name)
	img_name_pure = img_name.split("/")[-1]
	# outside raw_fames dir
	img_dir_dir = img_name.split("/")[:-2]
	img_dir_dir = ("/").join(img_dir_dir)
	# print(img_dir)
	crop_img = img[y:y+h, x:x+w]
	crop_img2 = img[y2:y2+h, x:x+w]
	cv2.imwrite(os.path.join(img_dir_dir, "rgb_frames", img_name_pure), crop_img)
	cv2.imwrite(os.path.join(img_dir_dir, "depth_frames", img_name_pure), crop_img2)
	

