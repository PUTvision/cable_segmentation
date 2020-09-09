import numpy as np
import segment
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch.transforms import ToTensorV2

scale_normalize = Compose([
    Resize(480, 640),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

model = segment.REMODEL_segmenter(data_path="dataset", batch_size=4, lr=3e-4).load_from_checkpoint("epoch=51.ckpt",
                                                                                                   batch_size=4,
                                                                                                   lr=3e-4)
input_img = cv2.imread("test.jpg")
vis_img = input_img.copy()
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_img = scale_normalize(image=input_img)['image']
input_img = input_img.float().view(-1, 3, 480, 640)
prediction = model(input_img)
prediction = (prediction.detach().numpy() * 255).astype(np.uint8).reshape(480, 640, 1)
prediction_overlay = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
temp = np.clip(vis_img.astype(np.int32) - prediction_overlay.astype(np.int32), 0, 255).astype(np.uint8)
prediction_overlay[:, :, 0] = 0
prediction_overlay[:, :, 2] = 0
final_overlay = (0.7 * vis_img + 0.3 * (temp + prediction_overlay)).astype(np.uint8)
stacked = np.hstack((vis_img, cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR), final_overlay))
cv2.imshow("win", stacked)
cv2.waitKey(0)
