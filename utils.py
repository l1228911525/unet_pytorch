import torch
import cv2
import numpy as np

device = torch.device("cpu")

def PgetImageAndLabel(inputPath, labelPath):
    Pmask = cv2.imread(labelPath[0], 0)
    Pmask = PresizeImg(Pmask)
    Pmask = np.array(Pmask)
    Pmask = Pmask[np.newaxis, :]

    input = cv2.imread(inputPath[0], 0)
    input = PresizeImg(input)
    input = np.array(input)
    input = input[np.newaxis, np.newaxis, :] / 255.0
    input = torch.tensor(input, dtype=torch.float32)

    for i in range(1, len(inputPath)):
        Pmask2 = cv2.imread(labelPath[i], 0)
        Pmask2 = PresizeImg(Pmask2)
        Pmask2 = np.array(Pmask2)
        Pmask2 = Pmask2[np.newaxis, :]

        input2 = cv2.imread(inputPath[i], 0)
        input2 = PresizeImg(input2)
        input2 = np.array(input2)
        input2 = input2[np.newaxis, np.newaxis, :] / 255.0
        input2 = torch.tensor(input2, dtype=torch.float32)

        Pmask = np.concatenate([Pmask, Pmask2], 0)
        input = torch.cat([input, input2], 0)
    input = input.to(device)
    return input, Pmask

def PresizeImg(img, stride = 16):
    H, W = img.shape
    dh = stride - (H % stride)
    dw = stride - (H % stride)
    img = cv2.copyMakeBorder(img, dh, 0, dw, 0, cv2.BORDER_CONSTANT, value=0)
    return img