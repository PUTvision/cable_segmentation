import cv2
import os

image_dir = "data/img"
mask_dir = "data/mask"
bg_dir = "data/bg"

img_list = sorted(os.listdir(image_dir))
bg_list = sorted(os.listdir(bg_dir))

target_dir = "dataset"


def get_only_object(img, mask, back_img):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask < 100] = 0
    fg = cv2.bitwise_or(img, img, mask=mask)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv[mask_inv < 100] = 0
    fg_back_inv = cv2.bitwise_or(back_img, back_img, mask=mask_inv)
    final = cv2.bitwise_or(fg, fg_back_inv)
    return final


for image_name in img_list:
    print(os.path.join(image_dir, image_name))
    image = cv2.imread(os.path.join(image_dir, image_name))
    mask = cv2.imread(os.path.join(mask_dir, image_name))
    name = os.path.splitext(image_name)[0]
    cv2.imwrite(os.path.join(target_dir, "img", name + "_" + "0".zfill(4) + ".jpg"), image)
    cv2.imwrite(os.path.join(target_dir, "msk", name + "_" + "0".zfill(4) + ".jpg"), mask)
    for index in range(0, len(bg_list) - 1):
        background = cv2.imread(os.path.join(bg_dir, bg_list[index]))
        # cv2.imshow("win", background)
        # cv2.waitKey(0)
        result = get_only_object(image, mask, background)
        cv2.imwrite(os.path.join(target_dir, "img", name + "_" + str(index + 1).zfill(4) + ".jpg"), result)
        cv2.imwrite(os.path.join(target_dir, "msk", name + "_" + str(index + 1).zfill(4) + ".jpg"), mask)
