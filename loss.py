import torch
import torch.nn as nn

device = torch.device("cpu")

def build_target(Pmask, PtypePixl = [0, 255]):
    Pmask = torch.from_numpy(Pmask)
    PmaskT = torch.zeros(size=(Pmask.shape[0], len(PtypePixl), Pmask.shape[1], Pmask.shape[2]))
    for b in range(Pmask.shape[0]):
        for i in range(len(PtypePixl)):
            PmaskT[b, i, ...][Pmask[b] == PtypePixl[i]] = 1
    return PmaskT

def PunetLoss(p, target, PtypePixl = [0, 255]):
    PbceLoss = nn.BCELoss(reduction="mean")
    PmaskT = build_target(target, PtypePixl)
    PmaskT = PmaskT.to(device)
    loss = PbceLoss(p, PmaskT)
    return loss