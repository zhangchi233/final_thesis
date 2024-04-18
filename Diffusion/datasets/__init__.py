from datasets.dataset import *

__all__ = ["LLdataset","DTUDataset","Dtu","TrainTestDtuDataset"]
from .dtu import DTUDataset,DTURefine,DTUDataset2


dataset_dict = {'dtu': DTUDataset,"DTUDataset":DTUDataset,"dtu2": DTUDataset2
               }