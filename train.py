import torch.optim as optim
import cv2
import numpy as np
from model import Net
import glob
import random
import torch
import utils
import loss

device = torch.device("cpu")

# model = Net()
PbatchSize = 1
model = torch.load("./unet.pkl", map_location="cpu")
model.train()
model = model.to(device)
PinputSet = glob.glob("./assets/train/*")
optimizer = optim.Adam(model.parameters(), lr=0.001)
for i in range(200001):
    index = random.randint(0, len(PinputSet) - PbatchSize - 1)
    inputPath = PinputSet[index : index + PbatchSize]
    labelPath = [x.replace("train", "label").replace(".tif", "_mask.tif") for x in inputPath]
    input, Pmask = utils.PgetImageAndLabel(inputPath, labelPath)

    if ((i%100) == 0) & (i != 0):
        torch.save(model, "./unet.pkl")
    optimizer.zero_grad()
    p = model(input)
    loss = loss.PunetLoss(p, Pmask)
    loss.backward()
    optimizer.step()
    print("loss:", loss)