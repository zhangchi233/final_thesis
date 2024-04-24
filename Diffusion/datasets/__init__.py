from datasets.dataset import *

__all__ = ["LLdataset","DTUDataset","Dtu","TrainTestDtuDataset"]
from .dtu import DTUDataset


dataset_dict = {'dtu': DTUDataset,"DTUDataset":DTUDataset,}