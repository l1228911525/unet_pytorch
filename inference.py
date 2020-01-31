import torch
import cv2
import numpy as np
from utils import PresizeImg

value = [0, 255]

img = cv2.imread("./assets/train/1_2.tif", 0)
# target = cv2.imread("./assetsCopy/label/1_2_mask.tif", 0)
# target = cv2.resize(target, (320, 320))
img = PresizeImg(img)
ori = img.copy()
input = np.array(img, dtype=np.float32)
input = input[np.newaxis, np.newaxis, :, :] / 255.0
model = torch.load("./unet.pkl", map_location=torch.device('cpu'))
model.eval()
output = model(torch.tensor(input, dtype=torch.float32))
print("out shape:", output.shape)
for i in range(output.shape[2]):
    for j in range(output.shape[3]):
        img[i, j] = value[torch.argmax(output[0, :, i, j])]

# cv2.imshow("target", target)
# cv2.imshow("ori", ori)
cv2.imwrite("img0.jpg", img)