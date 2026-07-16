import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, Dataset
from scipy.spatial import KDTree
import random
import copy
from concurrent.futures import ProcessPoolExecutor
import itertools
import open3d as o3d


# === SETTINGS ===
dataset_dir = "Datasets"
csv_files_training = [
    {"filename": "Eigenstrains_N001.csv", "boundary_file": "AllNodes_N001.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N001.csv", "boundary_file": "AllNodes_N001.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N002.csv", "boundary_file": "AllNodes_N002.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N002.csv", "boundary_file": "AllNodes_N002.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N003.csv", "boundary_file": "AllNodes_N003.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N003.csv", "boundary_file": "AllNodes_N003.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N004.csv", "boundary_file": "AllNodes_N004.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N004.csv", "boundary_file": "AllNodes_N004.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N005.csv", "boundary_file": "AllNodes_N005.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N005.csv", "boundary_file": "AllNodes_N005.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N006.csv", "boundary_file": "AllNodes_N006.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N006.csv", "boundary_file": "AllNodes_N006.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N007.csv", "boundary_file": "AllNodes_N007.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N007.csv", "boundary_file": "AllNodes_N007.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N008.csv", "boundary_file": "AllNodes_N008.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N008.csv", "boundary_file": "AllNodes_N008.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N009.csv", "boundary_file": "AllNodes_N009.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N009.csv", "boundary_file": "AllNodes_N009.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N010.csv", "boundary_file": "AllNodes_N010.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N010.csv", "boundary_file": "AllNodes_N010.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N011.csv", "boundary_file": "AllNodes_N011.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N011.csv", "boundary_file": "AllNodes_N011.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N012.csv", "boundary_file": "AllNodes_N012.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N012.csv", "boundary_file": "AllNodes_N012.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N013.csv", "boundary_file": "AllNodes_N013.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N013.csv", "boundary_file": "AllNodes_N013.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N014.csv", "boundary_file": "AllNodes_N014.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N014.csv", "boundary_file": "AllNodes_N014.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N015.csv", "boundary_file": "AllNodes_N015.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N015.csv", "boundary_file": "AllNodes_N015.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N016.csv", "boundary_file": "AllNodes_N016.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N016.csv", "boundary_file": "AllNodes_N016.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N017.csv", "boundary_file": "AllNodes_N017.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N017.csv", "boundary_file": "AllNodes_N017.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N018.csv", "boundary_file": "AllNodes_N018.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N018.csv", "boundary_file": "AllNodes_N018.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N019.csv", "boundary_file": "AllNodes_N019.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N019.csv", "boundary_file": "AllNodes_N019.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N020.csv", "boundary_file": "AllNodes_N020.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N020.csv", "boundary_file": "AllNodes_N020.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N021.csv", "boundary_file": "AllNodes_N021.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N021.csv", "boundary_file": "AllNodes_N021.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N022.csv", "boundary_file": "AllNodes_N022.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N022.csv", "boundary_file": "AllNodes_N022.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N023.csv", "boundary_file": "AllNodes_N023.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N023.csv", "boundary_file": "AllNodes_N023.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N024.csv", "boundary_file": "AllNodes_N024.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N024.csv", "boundary_file": "AllNodes_N024.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N025.csv", "boundary_file": "AllNodes_N025.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N025.csv", "boundary_file": "AllNodes_N025.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N026.csv", "boundary_file": "AllNodes_N026.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N026.csv", "boundary_file": "AllNodes_N026.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N027.csv", "boundary_file": "AllNodes_N027.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N027.csv", "boundary_file": "AllNodes_N027.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N028.csv", "boundary_file": "AllNodes_N028.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N028.csv", "boundary_file": "AllNodes_N028.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N029.csv", "boundary_file": "AllNodes_N029.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N029.csv", "boundary_file": "AllNodes_N029.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N030.csv", "boundary_file": "AllNodes_N030.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N030.csv", "boundary_file": "AllNodes_N030.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N031.csv", "boundary_file": "AllNodes_N031.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N031.csv", "boundary_file": "AllNodes_N031.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N032.csv", "boundary_file": "AllNodes_N032.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N032.csv", "boundary_file": "AllNodes_N032.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N033.csv", "boundary_file": "AllNodes_N033.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N033.csv", "boundary_file": "AllNodes_N033.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N034.csv", "boundary_file": "AllNodes_N034.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N034.csv", "boundary_file": "AllNodes_N034.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N035.csv", "boundary_file": "AllNodes_N035.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N035.csv", "boundary_file": "AllNodes_N035.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N036.csv", "boundary_file": "AllNodes_N036.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N036.csv", "boundary_file": "AllNodes_N036.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N037.csv", "boundary_file": "AllNodes_N037.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N037.csv", "boundary_file": "AllNodes_N037.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N038.csv", "boundary_file": "AllNodes_N038.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N038.csv", "boundary_file": "AllNodes_N038.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N039.csv", "boundary_file": "AllNodes_N039.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N039.csv", "boundary_file": "AllNodes_N039.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N040.csv", "boundary_file": "AllNodes_N040.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N040.csv", "boundary_file": "AllNodes_N040.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N041.csv", "boundary_file": "AllNodes_N041.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N041.csv", "boundary_file": "AllNodes_N041.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N042.csv", "boundary_file": "AllNodes_N042.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N042.csv", "boundary_file": "AllNodes_N042.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N043.csv", "boundary_file": "AllNodes_N043.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N043.csv", "boundary_file": "AllNodes_N043.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N044.csv", "boundary_file": "AllNodes_N044.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N044.csv", "boundary_file": "AllNodes_N044.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N045.csv", "boundary_file": "AllNodes_N045.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N045.csv", "boundary_file": "AllNodes_N045.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N046.csv", "boundary_file": "AllNodes_N046.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N046.csv", "boundary_file": "AllNodes_N046.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N047.csv", "boundary_file": "AllNodes_N047.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N047.csv", "boundary_file": "AllNodes_N047.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N048.csv", "boundary_file": "AllNodes_N048.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N048.csv", "boundary_file": "AllNodes_N048.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N049.csv", "boundary_file": "AllNodes_N049.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N049.csv", "boundary_file": "AllNodes_N049.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N050.csv", "boundary_file": "AllNodes_N050.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N050.csv", "boundary_file": "AllNodes_N050.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N051.csv", "boundary_file": "AllNodes_N051.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N051.csv", "boundary_file": "AllNodes_N051.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N052.csv", "boundary_file": "AllNodes_N052.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N052.csv", "boundary_file": "AllNodes_N052.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N053.csv", "boundary_file": "AllNodes_N053.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N053.csv", "boundary_file": "AllNodes_N053.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N054.csv", "boundary_file": "AllNodes_N054.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N054.csv", "boundary_file": "AllNodes_N054.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N055.csv", "boundary_file": "AllNodes_N055.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N055.csv", "boundary_file": "AllNodes_N055.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N056.csv", "boundary_file": "AllNodes_N056.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N056.csv", "boundary_file": "AllNodes_N056.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N057.csv", "boundary_file": "AllNodes_N057.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N057.csv", "boundary_file": "AllNodes_N057.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N058.csv", "boundary_file": "AllNodes_N058.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N058.csv", "boundary_file": "AllNodes_N058.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N059.csv", "boundary_file": "AllNodes_N059.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N059.csv", "boundary_file": "AllNodes_N059.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N060.csv", "boundary_file": "AllNodes_N060.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N060.csv", "boundary_file": "AllNodes_N060.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N061.csv", "boundary_file": "AllNodes_N061.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N061.csv", "boundary_file": "AllNodes_N061.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N062.csv", "boundary_file": "AllNodes_N062.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N062.csv", "boundary_file": "AllNodes_N062.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N063.csv", "boundary_file": "AllNodes_N063.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N063.csv", "boundary_file": "AllNodes_N063.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N064.csv", "boundary_file": "AllNodes_N064.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N064.csv", "boundary_file": "AllNodes_N064.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N065.csv", "boundary_file": "AllNodes_N065.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N065.csv", "boundary_file": "AllNodes_N065.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N066.csv", "boundary_file": "AllNodes_N066.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N066.csv", "boundary_file": "AllNodes_N066.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N067.csv", "boundary_file": "AllNodes_N067.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N067.csv", "boundary_file": "AllNodes_N067.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N068.csv", "boundary_file": "AllNodes_N068.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N068.csv", "boundary_file": "AllNodes_N068.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N069.csv", "boundary_file": "AllNodes_N069.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N069.csv", "boundary_file": "AllNodes_N069.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N070.csv", "boundary_file": "AllNodes_N070.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N070.csv", "boundary_file": "AllNodes_N070.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N071.csv", "boundary_file": "AllNodes_N071.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N071.csv", "boundary_file": "AllNodes_N071.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N072.csv", "boundary_file": "AllNodes_N072.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N072.csv", "boundary_file": "AllNodes_N072.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N073.csv", "boundary_file": "AllNodes_N073.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N073.csv", "boundary_file": "AllNodes_N073.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N074.csv", "boundary_file": "AllNodes_N074.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N074.csv", "boundary_file": "AllNodes_N074.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N075.csv", "boundary_file": "AllNodes_N075.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N075.csv", "boundary_file": "AllNodes_N075.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N076.csv", "boundary_file": "AllNodes_N076.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N076.csv", "boundary_file": "AllNodes_N076.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N077.csv", "boundary_file": "AllNodes_N077.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N077.csv", "boundary_file": "AllNodes_N077.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N078.csv", "boundary_file": "AllNodes_N078.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N078.csv", "boundary_file": "AllNodes_N078.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N079.csv", "boundary_file": "AllNodes_N079.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N079.csv", "boundary_file": "AllNodes_N079.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N080.csv", "boundary_file": "AllNodes_N080.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N080.csv", "boundary_file": "AllNodes_N080.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N081.csv", "boundary_file": "AllNodes_N081.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N081.csv", "boundary_file": "AllNodes_N081.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N082.csv", "boundary_file": "AllNodes_N082.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N082.csv", "boundary_file": "AllNodes_N082.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N083.csv", "boundary_file": "AllNodes_N083.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N083.csv", "boundary_file": "AllNodes_N083.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N084.csv", "boundary_file": "AllNodes_N084.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N084.csv", "boundary_file": "AllNodes_N084.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N085.csv", "boundary_file": "AllNodes_N085.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N085.csv", "boundary_file": "AllNodes_N085.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N086.csv", "boundary_file": "AllNodes_N086.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N086.csv", "boundary_file": "AllNodes_N086.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N087.csv", "boundary_file": "AllNodes_N087.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N087.csv", "boundary_file": "AllNodes_N087.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N088.csv", "boundary_file": "AllNodes_N088.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N088.csv", "boundary_file": "AllNodes_N088.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N089.csv", "boundary_file": "AllNodes_N089.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N089.csv", "boundary_file": "AllNodes_N089.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N090.csv", "boundary_file": "AllNodes_N090.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N090.csv", "boundary_file": "AllNodes_N090.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N091.csv", "boundary_file": "AllNodes_N091.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N091.csv", "boundary_file": "AllNodes_N091.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N092.csv", "boundary_file": "AllNodes_N092.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N092.csv", "boundary_file": "AllNodes_N092.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N093.csv", "boundary_file": "AllNodes_N093.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N093.csv", "boundary_file": "AllNodes_N093.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N094.csv", "boundary_file": "AllNodes_N094.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N094.csv", "boundary_file": "AllNodes_N094.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N095.csv", "boundary_file": "AllNodes_N095.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N095.csv", "boundary_file": "AllNodes_N095.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N096.csv", "boundary_file": "AllNodes_N096.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N096.csv", "boundary_file": "AllNodes_N096.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N097.csv", "boundary_file": "AllNodes_N097.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N097.csv", "boundary_file": "AllNodes_N097.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N098.csv", "boundary_file": "AllNodes_N098.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N098.csv", "boundary_file": "AllNodes_N098.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N099.csv", "boundary_file": "AllNodes_N099.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N099.csv", "boundary_file": "AllNodes_N099.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N100.csv", "boundary_file": "AllNodes_N100.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N100.csv", "boundary_file": "AllNodes_N100.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N101.csv", "boundary_file": "AllNodes_N101.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N101.csv", "boundary_file": "AllNodes_N101.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N102.csv", "boundary_file": "AllNodes_N102.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N102.csv", "boundary_file": "AllNodes_N102.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N103.csv", "boundary_file": "AllNodes_N103.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N103.csv", "boundary_file": "AllNodes_N103.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N104.csv", "boundary_file": "AllNodes_N104.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N104.csv", "boundary_file": "AllNodes_N104.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N105.csv", "boundary_file": "AllNodes_N105.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N105.csv", "boundary_file": "AllNodes_N105.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N106.csv", "boundary_file": "AllNodes_N106.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N106.csv", "boundary_file": "AllNodes_N106.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N107.csv", "boundary_file": "AllNodes_N107.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N107.csv", "boundary_file": "AllNodes_N107.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N108.csv", "boundary_file": "AllNodes_N108.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N108.csv", "boundary_file": "AllNodes_N108.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N109.csv", "boundary_file": "AllNodes_N109.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N109.csv", "boundary_file": "AllNodes_N109.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N110.csv", "boundary_file": "AllNodes_N110.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N110.csv", "boundary_file": "AllNodes_N110.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N111.csv", "boundary_file": "AllNodes_N111.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N111.csv", "boundary_file": "AllNodes_N111.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N112.csv", "boundary_file": "AllNodes_N112.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N112.csv", "boundary_file": "AllNodes_N112.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N113.csv", "boundary_file": "AllNodes_N113.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N113.csv", "boundary_file": "AllNodes_N113.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N114.csv", "boundary_file": "AllNodes_N114.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N114.csv", "boundary_file": "AllNodes_N114.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N115.csv", "boundary_file": "AllNodes_N115.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N115.csv", "boundary_file": "AllNodes_N115.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N116.csv", "boundary_file": "AllNodes_N116.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N116.csv", "boundary_file": "AllNodes_N116.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N117.csv", "boundary_file": "AllNodes_N117.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N117.csv", "boundary_file": "AllNodes_N117.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N118.csv", "boundary_file": "AllNodes_N118.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N118.csv", "boundary_file": "AllNodes_N118.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N119.csv", "boundary_file": "AllNodes_N119.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N119.csv", "boundary_file": "AllNodes_N119.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N120.csv", "boundary_file": "AllNodes_N120.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N120.csv", "boundary_file": "AllNodes_N120.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N121.csv", "boundary_file": "AllNodes_N121.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N121.csv", "boundary_file": "AllNodes_N121.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N122.csv", "boundary_file": "AllNodes_N122.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N122.csv", "boundary_file": "AllNodes_N122.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N123.csv", "boundary_file": "AllNodes_N123.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N123.csv", "boundary_file": "AllNodes_N123.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N124.csv", "boundary_file": "AllNodes_N124.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N124.csv", "boundary_file": "AllNodes_N124.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N125.csv", "boundary_file": "AllNodes_N125.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N125.csv", "boundary_file": "AllNodes_N125.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N126.csv", "boundary_file": "AllNodes_N126.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N126.csv", "boundary_file": "AllNodes_N126.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N127.csv", "boundary_file": "AllNodes_N127.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N127.csv", "boundary_file": "AllNodes_N127.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N128.csv", "boundary_file": "AllNodes_N128.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N128.csv", "boundary_file": "AllNodes_N128.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N129.csv", "boundary_file": "AllNodes_N129.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N129.csv", "boundary_file": "AllNodes_N129.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N130.csv", "boundary_file": "AllNodes_N130.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N130.csv", "boundary_file": "AllNodes_N130.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N131.csv", "boundary_file": "AllNodes_N131.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N131.csv", "boundary_file": "AllNodes_N131.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N132.csv", "boundary_file": "AllNodes_N132.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N132.csv", "boundary_file": "AllNodes_N132.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N133.csv", "boundary_file": "AllNodes_N133.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N133.csv", "boundary_file": "AllNodes_N133.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N134.csv", "boundary_file": "AllNodes_N134.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N134.csv", "boundary_file": "AllNodes_N134.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N135.csv", "boundary_file": "AllNodes_N135.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    # {"filename": "Eigenstrains_N135.csv", "boundary_file": "AllNodes_N135.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N136.csv", "boundary_file": "AllNodes_N136.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N136.csv", "boundary_file": "AllNodes_N136.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N137.csv", "boundary_file": "AllNodes_N137.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N137.csv", "boundary_file": "AllNodes_N137.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N138.csv", "boundary_file": "AllNodes_N138.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N138.csv", "boundary_file": "AllNodes_N138.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N139.csv", "boundary_file": "AllNodes_N139.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N139.csv", "boundary_file": "AllNodes_N139.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N140.csv", "boundary_file": "AllNodes_N140.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N140.csv", "boundary_file": "AllNodes_N140.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N141.csv", "boundary_file": "AllNodes_N141.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N141.csv", "boundary_file": "AllNodes_N141.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N142.csv", "boundary_file": "AllNodes_N142.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N142.csv", "boundary_file": "AllNodes_N142.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N143.csv", "boundary_file": "AllNodes_N143.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N143.csv", "boundary_file": "AllNodes_N143.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N144.csv", "boundary_file": "AllNodes_N144.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N144.csv", "boundary_file": "AllNodes_N144.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N145.csv", "boundary_file": "AllNodes_N145.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N145.csv", "boundary_file": "AllNodes_N145.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N146.csv", "boundary_file": "AllNodes_N146.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N146.csv", "boundary_file": "AllNodes_N146.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N147.csv", "boundary_file": "AllNodes_N147.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N147.csv", "boundary_file": "AllNodes_N147.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N148.csv", "boundary_file": "AllNodes_N148.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N148.csv", "boundary_file": "AllNodes_N148.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N149.csv", "boundary_file": "AllNodes_N149.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N149.csv", "boundary_file": "AllNodes_N149.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N150.csv", "boundary_file": "AllNodes_N150.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N150.csv", "boundary_file": "AllNodes_N150.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N151.csv", "boundary_file": "AllNodes_N151.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N151.csv", "boundary_file": "AllNodes_N151.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N152.csv", "boundary_file": "AllNodes_N152.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N152.csv", "boundary_file": "AllNodes_N152.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N153.csv", "boundary_file": "AllNodes_N153.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N153.csv", "boundary_file": "AllNodes_N153.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N154.csv", "boundary_file": "AllNodes_N154.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N154.csv", "boundary_file": "AllNodes_N154.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N155.csv", "boundary_file": "AllNodes_N155.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N155.csv", "boundary_file": "AllNodes_N155.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N156.csv", "boundary_file": "AllNodes_N156.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N156.csv", "boundary_file": "AllNodes_N156.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N157.csv", "boundary_file": "AllNodes_N157.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N157.csv", "boundary_file": "AllNodes_N157.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N158.csv", "boundary_file": "AllNodes_N158.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N158.csv", "boundary_file": "AllNodes_N158.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N159.csv", "boundary_file": "AllNodes_N159.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N159.csv", "boundary_file": "AllNodes_N159.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N160.csv", "boundary_file": "AllNodes_N160.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N160.csv", "boundary_file": "AllNodes_N160.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N161.csv", "boundary_file": "AllNodes_N161.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N161.csv", "boundary_file": "AllNodes_N161.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N162.csv", "boundary_file": "AllNodes_N162.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N162.csv", "boundary_file": "AllNodes_N162.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N163.csv", "boundary_file": "AllNodes_N163.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N163.csv", "boundary_file": "AllNodes_N163.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N164.csv", "boundary_file": "AllNodes_N164.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N164.csv", "boundary_file": "AllNodes_N164.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N165.csv", "boundary_file": "AllNodes_N165.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N165.csv", "boundary_file": "AllNodes_N165.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N166.csv", "boundary_file": "AllNodes_N166.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N166.csv", "boundary_file": "AllNodes_N166.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N167.csv", "boundary_file": "AllNodes_N167.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N167.csv", "boundary_file": "AllNodes_N167.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N168.csv", "boundary_file": "AllNodes_N168.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N168.csv", "boundary_file": "AllNodes_N168.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N169.csv", "boundary_file": "AllNodes_N169.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N169.csv", "boundary_file": "AllNodes_N169.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N170.csv", "boundary_file": "AllNodes_N170.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N170.csv", "boundary_file": "AllNodes_N170.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N171.csv", "boundary_file": "AllNodes_N171.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N171.csv", "boundary_file": "AllNodes_N171.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N172.csv", "boundary_file": "AllNodes_N172.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N172.csv", "boundary_file": "AllNodes_N172.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N173.csv", "boundary_file": "AllNodes_N173.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N173.csv", "boundary_file": "AllNodes_N173.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N174.csv", "boundary_file": "AllNodes_N174.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N174.csv", "boundary_file": "AllNodes_N174.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N175.csv", "boundary_file": "AllNodes_N175.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N175.csv", "boundary_file": "AllNodes_N175.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N176.csv", "boundary_file": "AllNodes_N176.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N176.csv", "boundary_file": "AllNodes_N176.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N177.csv", "boundary_file": "AllNodes_N177.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N177.csv", "boundary_file": "AllNodes_N177.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N178.csv", "boundary_file": "AllNodes_N178.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N178.csv", "boundary_file": "AllNodes_N178.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N179.csv", "boundary_file": "AllNodes_N179.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N179.csv", "boundary_file": "AllNodes_N179.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N180.csv", "boundary_file": "AllNodes_N180.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N180.csv", "boundary_file": "AllNodes_N180.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N181.csv", "boundary_file": "AllNodes_N181.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N181.csv", "boundary_file": "AllNodes_N181.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N182.csv", "boundary_file": "AllNodes_N182.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N182.csv", "boundary_file": "AllNodes_N182.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N183.csv", "boundary_file": "AllNodes_N183.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N183.csv", "boundary_file": "AllNodes_N183.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N184.csv", "boundary_file": "AllNodes_N184.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N184.csv", "boundary_file": "AllNodes_N184.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N185.csv", "boundary_file": "AllNodes_N185.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N185.csv", "boundary_file": "AllNodes_N185.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N186.csv", "boundary_file": "AllNodes_N186.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N186.csv", "boundary_file": "AllNodes_N186.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N187.csv", "boundary_file": "AllNodes_N187.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N187.csv", "boundary_file": "AllNodes_N187.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N188.csv", "boundary_file": "AllNodes_N188.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N188.csv", "boundary_file": "AllNodes_N188.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N189.csv", "boundary_file": "AllNodes_N189.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N189.csv", "boundary_file": "AllNodes_N189.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N190.csv", "boundary_file": "AllNodes_N190.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N190.csv", "boundary_file": "AllNodes_N190.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N191.csv", "boundary_file": "AllNodes_N191.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N191.csv", "boundary_file": "AllNodes_N191.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N192.csv", "boundary_file": "AllNodes_N192.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N192.csv", "boundary_file": "AllNodes_N192.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N193.csv", "boundary_file": "AllNodes_N193.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N193.csv", "boundary_file": "AllNodes_N193.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N194.csv", "boundary_file": "AllNodes_N194.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N194.csv", "boundary_file": "AllNodes_N194.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N195.csv", "boundary_file": "AllNodes_N195.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N195.csv", "boundary_file": "AllNodes_N195.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N196.csv", "boundary_file": "AllNodes_N196.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N196.csv", "boundary_file": "AllNodes_N196.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N197.csv", "boundary_file": "AllNodes_N197.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N197.csv", "boundary_file": "AllNodes_N197.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N198.csv", "boundary_file": "AllNodes_N198.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N198.csv", "boundary_file": "AllNodes_N198.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N199.csv", "boundary_file": "AllNodes_N199.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N199.csv", "boundary_file": "AllNodes_N199.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N200.csv", "boundary_file": "AllNodes_N200.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N200.csv", "boundary_file": "AllNodes_N200.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N201.csv", "boundary_file": "AllNodes_N201.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N201.csv", "boundary_file": "AllNodes_N201.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N202.csv", "boundary_file": "AllNodes_N202.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N202.csv", "boundary_file": "AllNodes_N202.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N203.csv", "boundary_file": "AllNodes_N203.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N203.csv", "boundary_file": "AllNodes_N203.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N204.csv", "boundary_file": "AllNodes_N204.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N204.csv", "boundary_file": "AllNodes_N204.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N205.csv", "boundary_file": "AllNodes_N205.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N205.csv", "boundary_file": "AllNodes_N205.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N206.csv", "boundary_file": "AllNodes_N206.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N206.csv", "boundary_file": "AllNodes_N206.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N207.csv", "boundary_file": "AllNodes_N207.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N207.csv", "boundary_file": "AllNodes_N207.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N208.csv", "boundary_file": "AllNodes_N208.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N208.csv", "boundary_file": "AllNodes_N208.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N209.csv", "boundary_file": "AllNodes_N209.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N209.csv", "boundary_file": "AllNodes_N209.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N210.csv", "boundary_file": "AllNodes_N210.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N210.csv", "boundary_file": "AllNodes_N210.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N211.csv", "boundary_file": "AllNodes_N211.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N211.csv", "boundary_file": "AllNodes_N211.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N212.csv", "boundary_file": "AllNodes_N212.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N212.csv", "boundary_file": "AllNodes_N212.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N213.csv", "boundary_file": "AllNodes_N213.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N213.csv", "boundary_file": "AllNodes_N213.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N214.csv", "boundary_file": "AllNodes_N214.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N214.csv", "boundary_file": "AllNodes_N214.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N215.csv", "boundary_file": "AllNodes_N215.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N215.csv", "boundary_file": "AllNodes_N215.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N216.csv", "boundary_file": "AllNodes_N216.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N216.csv", "boundary_file": "AllNodes_N216.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N217.csv", "boundary_file": "AllNodes_N217.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N217.csv", "boundary_file": "AllNodes_N217.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N218.csv", "boundary_file": "AllNodes_N218.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N218.csv", "boundary_file": "AllNodes_N218.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N219.csv", "boundary_file": "AllNodes_N219.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N219.csv", "boundary_file": "AllNodes_N219.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N220.csv", "boundary_file": "AllNodes_N220.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N220.csv", "boundary_file": "AllNodes_N220.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N221.csv", "boundary_file": "AllNodes_N221.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N221.csv", "boundary_file": "AllNodes_N221.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N222.csv", "boundary_file": "AllNodes_N222.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N222.csv", "boundary_file": "AllNodes_N222.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N223.csv", "boundary_file": "AllNodes_N223.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N223.csv", "boundary_file": "AllNodes_N223.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N224.csv", "boundary_file": "AllNodes_N224.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N224.csv", "boundary_file": "AllNodes_N224.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N225.csv", "boundary_file": "AllNodes_N225.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N225.csv", "boundary_file": "AllNodes_N225.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N226.csv", "boundary_file": "AllNodes_N226.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N226.csv", "boundary_file": "AllNodes_N226.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N227.csv", "boundary_file": "AllNodes_N227.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N227.csv", "boundary_file": "AllNodes_N227.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N228.csv", "boundary_file": "AllNodes_N228.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N228.csv", "boundary_file": "AllNodes_N228.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N229.csv", "boundary_file": "AllNodes_N229.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N229.csv", "boundary_file": "AllNodes_N229.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N230.csv", "boundary_file": "AllNodes_N230.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N230.csv", "boundary_file": "AllNodes_N230.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N231.csv", "boundary_file": "AllNodes_N231.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N231.csv", "boundary_file": "AllNodes_N231.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N232.csv", "boundary_file": "AllNodes_N232.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N232.csv", "boundary_file": "AllNodes_N232.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N233.csv", "boundary_file": "AllNodes_N233.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N233.csv", "boundary_file": "AllNodes_N233.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N234.csv", "boundary_file": "AllNodes_N234.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N234.csv", "boundary_file": "AllNodes_N234.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N235.csv", "boundary_file": "AllNodes_N235.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N235.csv", "boundary_file": "AllNodes_N235.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N236.csv", "boundary_file": "AllNodes_N236.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N236.csv", "boundary_file": "AllNodes_N236.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N237.csv", "boundary_file": "AllNodes_N237.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N237.csv", "boundary_file": "AllNodes_N237.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N238.csv", "boundary_file": "AllNodes_N238.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N238.csv", "boundary_file": "AllNodes_N238.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N239.csv", "boundary_file": "AllNodes_N239.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N239.csv", "boundary_file": "AllNodes_N239.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N240.csv", "boundary_file": "AllNodes_N240.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N240.csv", "boundary_file": "AllNodes_N240.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N241.csv", "boundary_file": "AllNodes_N241.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N241.csv", "boundary_file": "AllNodes_N241.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N242.csv", "boundary_file": "AllNodes_N242.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N242.csv", "boundary_file": "AllNodes_N242.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N243.csv", "boundary_file": "AllNodes_N243.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N243.csv", "boundary_file": "AllNodes_N243.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N244.csv", "boundary_file": "AllNodes_N244.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N244.csv", "boundary_file": "AllNodes_N244.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N245.csv", "boundary_file": "AllNodes_N245.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N245.csv", "boundary_file": "AllNodes_N245.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N246.csv", "boundary_file": "AllNodes_N246.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N246.csv", "boundary_file": "AllNodes_N246.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N247.csv", "boundary_file": "AllNodes_N247.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N247.csv", "boundary_file": "AllNodes_N247.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N248.csv", "boundary_file": "AllNodes_N248.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N248.csv", "boundary_file": "AllNodes_N248.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N249.csv", "boundary_file": "AllNodes_N249.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N249.csv", "boundary_file": "AllNodes_N249.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N250.csv", "boundary_file": "AllNodes_N250.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N250.csv", "boundary_file": "AllNodes_N250.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N251.csv", "boundary_file": "AllNodes_N251.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N251.csv", "boundary_file": "AllNodes_N251.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N252.csv", "boundary_file": "AllNodes_N252.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N252.csv", "boundary_file": "AllNodes_N252.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N253.csv", "boundary_file": "AllNodes_N253.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N253.csv", "boundary_file": "AllNodes_N253.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N254.csv", "boundary_file": "AllNodes_N254.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N254.csv", "boundary_file": "AllNodes_N254.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N255.csv", "boundary_file": "AllNodes_N255.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N255.csv", "boundary_file": "AllNodes_N255.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N256.csv", "boundary_file": "AllNodes_N256.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N256.csv", "boundary_file": "AllNodes_N256.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N257.csv", "boundary_file": "AllNodes_N257.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N257.csv", "boundary_file": "AllNodes_N257.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N258.csv", "boundary_file": "AllNodes_N258.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N258.csv", "boundary_file": "AllNodes_N258.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N259.csv", "boundary_file": "AllNodes_N259.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N259.csv", "boundary_file": "AllNodes_N259.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N260.csv", "boundary_file": "AllNodes_N260.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_N260.csv", "boundary_file": "AllNodes_N260.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},

]

csv_files_validation = [
{"filename": "Eigenstrains_N092.csv", "boundary_file": "AllNodes_N092.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
{"filename": "Eigenstrains_N092.csv", "boundary_file": "AllNodes_N092.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
{"filename": "Eigenstrains_N135.csv", "boundary_file": "AllNodes_N135.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
{"filename": "Eigenstrains_N135.csv", "boundary_file": "AllNodes_N135.csv","Rot": True, "X_trans": 0, "Y_trans":0, "Z_trans":0},
 
]

csv_files_test = [
    {"filename": "Eigenstrains_CL12.csv", "boundary_file": "AllNodes_CL12.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_CL24.csv", "boundary_file": "AllNodes_CL24.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_CL36.csv", "boundary_file": "AllNodes_CL36.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_CL48.csv", "boundary_file": "AllNodes_CL48.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_CL60.csv", "boundary_file": "AllNodes_CL60.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_CL72.csv", "boundary_file": "AllNodes_CL72.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_CL84.csv", "boundary_file": "AllNodes_CL84.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_CL96.csv", "boundary_file": "AllNodes_CL96.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_ILLGeometry.csv", "boundary_file": "AllNodes_ILLGeometry.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    {"filename": "Eigenstrains_ISISGeometry.csv", "boundary_file": "AllNodes_ISISGeometry.csv","Rot": False, "X_trans": 0, "Y_trans":0, "Z_trans":0},
    
 
]





# === Initialize the master dictionary ===
all_data_original = {
    'vertices': [],
    'nodal_eigenstrain': [],
    'points_cloud': [],
    'file_name': [],
    'w': [],
    'L': []
}

# === DATA PREPROCESSING FUNCTION ===
def data_prep(COORDS_raw, Eigenstrains_raw, boundary_raw, Rot, X_trans, Y_trans, Z_trans):
    
    #boundary_raw(N,5): NodeID|X|Y|Z|boundaryflag
    #COORDS(N,7): ElementID|X|Y|Z|X_centroid|Y_centroid|Z_centroid|
    #Eigenstrains(N,6)
    
    boundary = boundary_raw[boundary_raw['boundaryflag'] == 1]
    
    # Ensure inputs are NumPy arrays, create overhang and boundary
    COORDS_raw = np.array(COORDS_raw)
    boundary = np.array(boundary) #only boundary nodes 
    boundary_raw = np.array(boundary_raw) #boundary and interior nodes
    overhang_raw = np.array(boundary_raw) #boundary and interior nodes
    overhang_raw[:,4]=0 #reset boundary for overhang points
    overhang = overhang_raw[overhang_raw[:, 4] == 1]
    
    print(boundary_raw.shape)
    print(boundary.shape)
    print(overhang.shape)
    

    
    #Apply 180 degree rotation if Rot=True
    if Rot:
        COORDS_raw[:, 1] *= -1
        COORDS_raw[:, 2] *= -1
        COORDS_raw[:, 4] *= -1
        COORDS_raw[:, 5] *= -1
        boundary_raw[:, 1] *= -1
        boundary_raw[:, 2] *= -1
        boundary[:, 1] *= -1
        boundary[:, 2] *= -1
        overhang_raw[:, 1] *= -1
        overhang_raw[:, 2] *= -1
        overhang[:, 1] *= -1
        overhang[:, 2] *= -1
        Eigenstrains_raw[:,4] *= -1
        Eigenstrains_raw[:,5] *= -1
    
       
    
    # Modify Boundary to get the true boundary
    tol = 0.01
    Z = boundary[:, 3]
    # 1. Find indices of the two planes
    mask_z06 = np.abs(Z - 0.6) < tol
    mask_z12 = np.abs(Z - 1.2) < tol
    # 2. Shift both planes down by 0.6
    boundary_shifted = boundary.copy()
    boundary_shifted[mask_z06, 3] -= 0.6
    boundary_shifted[mask_z12, 3] -= 0.6
    # 3. Take second plane (originally Z~1.2, now at Z~0.6), duplicate and move to Z=1.2
    plane_copy = boundary_shifted[mask_z12].copy()
    plane_copy[:, 3] = 1.2  # set Z to 1.2
    # 4. Append the duplicated plane
    print(boundary.shape)
    boundary = np.vstack([boundary_shifted, plane_copy])
    print(boundary.shape)
    
    
    
    # Split between AM and BP
    mask = COORDS_raw[:, 3] > 0
    COORDS_AM = COORDS_raw[mask]
    COORDS_BP = COORDS_raw[~mask]
    Eigenstrains_AM = Eigenstrains_raw[mask]
    Eigenstrains_BP = Eigenstrains_raw[~mask]
    idx_AM = np.where(mask)[0]   # index to get the original order at the end during stacking
    idx_BP = np.where(~mask)[0]

      
    
    # === Local border distance LOGIC ===
    
    # Extract coordinates from COORDS
    coords_xyzCxCy = COORDS_AM[:, 1:6]  # X, Y, Z (columns 1-3)
    
    # Extract boundary XYZ positions
    boundary_xyz = boundary[:, 1:4]  # X, Y, Z (columns 1-3)
    
    tol = 0.6  # Tolerance
    step_size = 0.6   # Step length
    step_eps = 0.05   # Allowed deviation
    
    min_x_pos = []
    min_x_neg = []
    min_y_pos = []
    min_y_neg = []
    min_z_pos = []
    min_z_neg = []
    
    min_Cx_pos = []
    min_Cx_neg = []
    min_Cy_pos = []
    min_Cy_neg = []
    
    Rx = []
    Ry = []
    Rz = []
    
    for point in coords_xyzCxCy:
        x, y, z, Cx, Cy = point
        
        # --- True Euclidean radial distance to boundary ---
        if boundary_xyz.size > 0:
            # Compute the vector differences from the current point to all boundary points
            deltas = boundary_xyz - np.array([x, y, z])  # Shape (N_boundary, 3)
        
            # Compute squared Euclidean distances
            squared_dists = np.sum(deltas**2, axis=1)
        
            # Get index of the closest boundary point
            min_idx = np.argmin(squared_dists)
            dx, dy, dz = deltas[min_idx]  # Corresponding dx, dy, dz
        
            # Append components of radius vector
            Rx.append(dx)
            Ry.append(dy)
            Rz.append(dz)
        else:
            Rx.append(np.nan)
            Ry.append(np.nan)
            Rz.append(np.nan)
    
    
    
        # --- Cartesian distance to boundary ---
        ### X direction (fix Y and Z)
        yz_mask = (
            (np.abs(boundary_xyz[:, 1] - y) <= tol) &
            (np.abs(boundary_xyz[:, 2] - z) <= tol)
        )
        yz_neighbors = boundary_xyz[yz_mask]
        dx_all = yz_neighbors[:, 0] - x if yz_neighbors.size > 0 else np.array([])
    
        # --- Positive X direction ---
        if dx_all.size > 0:
            dx_pos = dx_all[dx_all > 0]
            if dx_pos.size > 0:
                dx = np.min(dx_pos)
                target = x + dx
                dCx = target - Cx
    
                # Step forward
                while True:
                    next_target = target + step_size
                    mask = np.abs(yz_neighbors[:, 0] - next_target) <= step_eps
                    if np.any(mask):
                        target = yz_neighbors[mask][:, 0][0]  # Step to new point
                        dx = target - x
                        dCx = target - Cx
                    else:
                        break
                min_x_pos.append(dx)
                min_Cx_pos.append(dCx)
            else:
                min_x_pos.append(np.nan)
                min_Cx_pos.append(np.nan)
        else:
            min_x_pos.append(np.nan)
            min_Cx_pos.append(np.nan)
    
        # --- Negative X direction ---
        if dx_all.size > 0:
            dx_neg = dx_all[dx_all < 0]
            if dx_neg.size > 0:
                dx = np.max(dx_neg)
                target = x + dx
                dCx = target - Cx
    
                # Step backward
                while True:
                    next_target = target - step_size
                    mask = np.abs(yz_neighbors[:, 0] - next_target) <= step_eps
                    if np.any(mask):
                        target = yz_neighbors[mask][:, 0][0]
                        dx = target - x
                        dCx = target - Cx
                    else:
                        break
                min_x_neg.append(dx)
                min_Cx_neg.append(dCx)
            else:
                min_x_neg.append(np.nan)
                min_Cx_neg.append(np.nan)
        else:
            min_x_neg.append(np.nan)
            min_Cx_neg.append(np.nan)
    
        ### Y direction (fix X and Z)
        xz_mask = (
            (np.abs(boundary_xyz[:, 0] - x) <= tol) &
            (np.abs(boundary_xyz[:, 2] - z) <= tol)
        )
        xz_neighbors = boundary_xyz[xz_mask]
        dy_all = xz_neighbors[:, 1] - y if xz_neighbors.size > 0 else np.array([])
    
        # Positive Y
        if dy_all.size > 0:
            dy_pos = dy_all[dy_all > 0]
            if dy_pos.size > 0:
                dy = np.min(dy_pos)
                target = y + dy
                dCy = target - Cy
                while True:
                    next_target = target + step_size
                    mask = np.abs(xz_neighbors[:, 1] - next_target) <= step_eps
                    if np.any(mask):
                        target = xz_neighbors[mask][:, 1][0]
                        dy = target - y
                        dCy = target - Cy
                    else:
                        break
                min_y_pos.append(dy)
                min_Cy_pos.append(dCy)
            else:
                min_y_pos.append(np.nan)
                min_Cy_pos.append(np.nan)
        else:
            min_y_pos.append(np.nan)
            min_Cy_pos.append(np.nan)
    
        # Negative Y
        if dy_all.size > 0:
            dy_neg = dy_all[dy_all < 0]
            if dy_neg.size > 0:
                dy = np.max(dy_neg)
                target = y + dy
                dCy = target - Cy
                while True:
                    next_target = target - step_size
                    mask = np.abs(xz_neighbors[:, 1] - next_target) <= step_eps
                    if np.any(mask):
                        target = xz_neighbors[mask][:, 1][0]
                        dy = target - y
                        dCy = target - Cy
                    else:
                        break
                min_y_neg.append(dy)
                min_Cy_neg.append(dCy)
            else:
                min_y_neg.append(np.nan)
                min_Cy_neg.append(np.nan)
        else:
            min_y_neg.append(np.nan)
            min_Cy_neg.append(np.nan)
            
            
        ### Z direction (fix X and Y)
        xy_mask = (
            (np.abs(boundary_xyz[:, 0] - x) <= tol) &
            (np.abs(boundary_xyz[:, 1] - y) <= tol)
        )
        xy_neighbors = boundary_xyz[xy_mask]
        dz_all = xy_neighbors[:, 2] - z if xy_neighbors.size > 0 else np.array([])
    
        # Positive Z
        if dz_all.size > 0:
            dz_pos = dz_all[dz_all > 0]
            if dz_pos.size > 0:
                dz = np.min(dz_pos)
                target = z + dz
                while True:
                    next_target = target + step_size
                    mask = np.abs(xy_neighbors[:, 2] - next_target) <= step_eps
                    if np.any(mask):
                        target = xy_neighbors[mask][:, 2][0]
                        dz = target - z
                    else:
                        break
                min_z_pos.append(dz)
            else:
                min_z_pos.append(np.nan)
        else:
            min_z_pos.append(np.nan)
    
        # Negative Z
        if dz_all.size > 0:
            dz_neg = dz_all[dz_all < 0]
            if dz_neg.size > 0:
                dz = np.max(dz_neg)
                target = z + dz
                while True:
                    next_target = target - step_size
                    mask = np.abs(xy_neighbors[:, 2] - next_target) <= step_eps
                    if np.any(mask):
                        target = xy_neighbors[mask][:, 2][0]
                        dz = target - z
                    else:
                        break
                min_z_neg.append(dz)
            else:
                min_z_neg.append(np.nan)
        else:
            min_z_neg.append(np.nan)
        
    
    # === END local border distance LOGIC ===
    
    # === IP Creation ===
    #Create IP relative position to centroid in terms of boundary distance  
    # Compute smallest absolute distances (pos and neg direction) for IP coordinates
    min_x_point = np.minimum(np.abs(min_x_pos), np.abs(min_x_neg))
    min_y_point = np.minimum(np.abs(min_y_pos), np.abs(min_y_neg))
    
    # Compute smallest absolute distances (pos and neg direction) for centroids
    min_x_centroid = np.minimum(np.abs(min_Cx_pos), np.abs(min_Cx_neg))
    min_y_centroid = np.minimum(np.abs(min_Cy_pos), np.abs(min_Cy_neg))
    
    # Compute IP as the difference
    IP_x = min_x_point - min_x_centroid
    IP_y = min_y_point - min_y_centroid
    IP_z = COORDS_AM[:,3] - COORDS_AM[:,6]
    
    # === END IP LOGIC ===
    
    
    # === OVERHANG DETECTION AND GRADIENT ASSIGNMENT ===

    boundary_xyz = boundary[:, 1:4]  # X, Y, Z
    coords_z = COORDS_AM[:, 3]

    x_min = np.min(boundary_xyz[:, 0])
    x_max = np.max(boundary_xyz[:, 0])
    y_min = np.min(boundary_xyz[:, 1])
    y_max = np.max(boundary_xyz[:, 1])
    z_min = np.min(boundary_xyz[:, 2])
    z_max = np.max(boundary_xyz[:, 2])
    
    # Append top and bottom to overhang already
    ind_min = np.where(np.abs(overhang_raw[:, 3] - z_min) < tol)[0]
    ind_max = np.where(np.abs(overhang_raw[:, 3] - z_max) < tol)[0]
    overhang_raw[ind_min,4]=1
    overhang_raw[ind_max,4]=1
    overhang = overhang_raw[overhang_raw[:, 4] == 1]
    
    tol = 0.01
    step = 0.6
    xy_tol = 0.15
    neighbor_radius = 0.85

    coords_x = COORDS_AM[:, 1]
    coords_y = COORDS_AM[:, 2]

    # List to collect overhang z levels
    overhang_points = []

    for i, (x, y, z) in enumerate(boundary_xyz):
        # Skip all bottom, top and lateral boundary points
        if x <= x_min + tol or x >= x_max - tol:
            continue
        if y <= y_min + tol or y >= y_max - tol:
            continue
        if z <= z_min + tol or z >= z_max - tol:
            continue

        # Condition 1: No boundary node approx 0.6 ± tol below this node
        below_mask = (
            (np.abs(boundary_xyz[:, 0] - x) <= tol) &
            (np.abs(boundary_xyz[:, 1] - y) <= tol) &
            (boundary_xyz[:, 2] >= z - step - tol) &
            (boundary_xyz[:, 2] <= z - step + tol)
        )
        if np.any(below_mask):
            continue  # boundary below → no overhang
            
        # Condition 2 (new): Must have lateral neighbor in XY within neighbor_radius
        dx = boundary_xyz[:, 0] - x
        dy = boundary_xyz[:, 1] - y
        radial_dist = np.sqrt(dx**2 + dy**2)
        
        lateral_neighbors_mask = (
            (radial_dist > 0.61) & (radial_dist < 0.9) &
            (np.abs(boundary_xyz[:, 2] - z) <= tol)
        )
        
        if np.sum(lateral_neighbors_mask) < 2:
            continue  # less than 2 lateral neighbors → no overhang
    
        # Condition 3: No COORDS Z in [z-0.6, z)
        coord_mask = (
            (coords_z >= z - step) & (coords_z < z) &
            (coords_x >= x - xy_tol) & (coords_x <= x + xy_tol) &
            (coords_y >= y - xy_tol) & (coords_y <= y + xy_tol)
        )
        
        if np.any(coord_mask):
            continue  # integration point below in neighborhood → no overhang

        # Passed conditions → this z is an overhang level
        overhang_points.append((z, x, y))

    # Optional: keep unique z values only
    seen_z = set()
    unique_overhang_points = []
    for z, x, y in overhang_points:
        if z not in seen_z:
            unique_overhang_points.append((z, x, y))
            seen_z.add(z)
    overhang_points = unique_overhang_points
    
    # Define representative x, y for z_min and z_max (e.g., domain center)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    

    # If no overhangs detected, create an empty array for gradient assignment
    if len(overhang_points) == 0:
        z_fallback = np.min(boundary[:, 3])
        x_fallback = (x_min + x_max) / 2
        y_fallback = (y_min + y_max) / 2
        overhang_points = [(z_fallback, x_fallback, y_fallback)]

    # Use min_x/y_pos/neg from your previous calculations to limit domain
    # Pick one COORDS point closest to each overhang z, get its limits
    min_x_pos_arr = np.array(min_x_pos)
    min_x_neg_arr = np.array(min_x_neg)
    min_y_pos_arr = np.array(min_y_pos)
    min_y_neg_arr = np.array(min_y_neg)
    
    print(overhang_points)

    overhang_gradient = np.zeros_like(coords_z)

    for z_oh, x_oh, y_oh in overhang_points:
        # Define the vertical range [z_oh + 0.6, z_oh + 2.4]
        z_start = z_oh + step
        z_end = z_oh + 4 * step  # 2.4 = 4 * 0.6

        z_mask = (coords_z >= z_start) & (coords_z <= z_end)
        xy_mask = (
            (np.abs(coords_x - x_oh) < xy_tol) &
            (np.abs(coords_y - y_oh) < xy_tol)
        )
        combined_mask = z_mask & xy_mask
        
        if not np.any(combined_mask):
            continue
        
        idx_closest = np.argmin(np.abs(coords_z[combined_mask] - z_end))
        idx_global = np.arange(len(coords_z))[combined_mask][idx_closest]

        # Extract min_x/y pos/neg for this point as limits
        x_pos_lim = min_x_pos_arr[idx_global] + coords_x[idx_global] + 2*tol
        x_neg_lim = min_x_neg_arr[idx_global] + coords_x[idx_global] - 2*tol
        y_pos_lim = min_y_pos_arr[idx_global] + coords_y[idx_global] + 2*tol
        y_neg_lim = min_y_neg_arr[idx_global] + coords_y[idx_global] - 2*tol
        
        
        # Now assign gradient values for COORDS within z range AND within x/y limits
        for i_pt, (x_pt, y_pt, z_pt) in enumerate(COORDS_AM[:, 1:4]):
            if z_pt < z_start or z_pt > z_end:
                continue

            # Check x limits
            if (x_pt > x_pos_lim) or (x_pt < x_neg_lim):
                continue

            # Check y limits
            if (y_pt > y_pos_lim) or (y_pt < y_neg_lim):
                continue

            # Calculate gradient value from 1 at z_start down to 0 at z_end
            gradient = 1 - (z_pt - z_start) / (z_end - z_start)
            gradient = np.clip(gradient, 0, 1)
            overhang_gradient[i_pt] = max(overhang_gradient[i_pt], gradient)
            
            

        # Step 3: Now select nodes in boundary_raw within [z_oh + step ± tol] and x/y limits
        z_start = z_oh + step - tol
        z_end = z_oh + step + tol

        for i in range(overhang_raw.shape[0]):
            x_b, y_b, z_b = overhang_raw[i, 1:4]

            if not (z_start <= z_b <= z_end):
                continue
            if (x_b > x_pos_lim) or (x_b < x_neg_lim):
                continue
            if (y_b > y_pos_lim) or (y_b < y_neg_lim):
                continue

            overhang_raw[i, 4] = 1  # Set boundary flag = 1
            print(i,x_b, y_b, z_b,x_pos_lim,x_neg_lim,y_pos_lim,y_neg_lim)

        # Step 4: Reconstruct boundary array from updated boundary_raw
        overhang = overhang_raw[overhang_raw[:, 4] == 1]
        print(boundary.shape)
        print(overhang.shape)

    # === END OVERHANG LOGIC ===
    
    
    # === REDO z min max dist including overhangboundary ===
    #CONTINUE HERE HERE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    min_z_pos2 = []
    min_z_neg2 = []
    
    # Extract boundary XYZ positions
    overhang_xyz = overhang[:, 1:4]  # X, Y, Z (columns 1-3)
    
    tol = 0.3  # Tolerance
    step_size = 0.6   # Step length
    step_eps = 0.01   # Allowed deviation
    for point in coords_xyzCxCy:
        x, y, z, Cx, Cy = point
        
            
        ### Z direction (fix X and Y)
        xy_mask = (
            (np.abs(overhang_xyz[:, 0] - x) <= tol) &
            (np.abs(overhang_xyz[:, 1] - y) <= tol)
        )
        xy_neighbors = overhang_xyz[xy_mask]
        dz_all = xy_neighbors[:, 2] - z if xy_neighbors.size > 0 else np.array([])
    
        # Positive Z
        if dz_all.size > 0:
            dz_pos = dz_all[dz_all > 0]
            if dz_pos.size > 0:
                dz = np.min(dz_pos)
                target = z + dz
                while True:
                    next_target = target + step_size
                    mask = np.abs(xy_neighbors[:, 2] - next_target) <= step_eps
                    if np.any(mask):
                        target = xy_neighbors[mask][:, 2][0]
                        dz = target - z
                    else:
                        break
                min_z_pos2.append(dz)
            else:
                min_z_pos2.append(np.nan)
                if np.isnan(min_z_pos2[-1]):
                    print("The last pos value is NaN")
        else:
            min_z_pos2.append(np.nan)
            if np.isnan(min_z_pos2[-1]):
                print("The last pos value is NaN")
    
        # Negative Z
        if dz_all.size > 0:
            dz_neg = dz_all[dz_all < 0]
            if dz_neg.size > 0:
                dz = np.max(dz_neg)
                target = z + dz
                while True:
                    next_target = target - step_size
                    mask = np.abs(xy_neighbors[:, 2] - next_target) <= step_eps
                    if np.any(mask):
                        target = xy_neighbors[mask][:, 2][0]
                        dz = target - z
                    else:
                        break
                min_z_neg2.append(dz)
            else:
                min_z_neg2.append(np.nan)
                if np.isnan(min_z_neg2[-1]):
                    print("The last neg value is NaN")
        else:
            min_z_neg2.append(np.nan)
            if np.isnan(min_z_neg2[-1]):
                print("The last neg value is NaN")
            
    # take the smaller min_z_pos /_neg (one fron pure boundary points, one from overhang)
    # use np.fmin for the case one value is nan
    # Positive region
    min_z_pos = np.fmin(min_z_pos, min_z_pos2)
    # Negative region: take min of abs value, i.e. max to get the one closer to zero
    min_z_neg = np.fmax(min_z_neg, min_z_neg2)
    # === END local border distance LOGIC ===

    #Compute 2D radius in zx plane
    R2x_xpos = []
    R2x_xneg = []
    R2z_xpos = []
    R2z_xneg = []
    
    tol = 0.3     # tolerance for matching Y values
    tol2 = min(min_y_pos)*2    # threshold for using modified y-plane
    factor = 4.73194  # scaling factor
    
    for i, point in enumerate(coords_xyzCxCy):
        x, y, z, Cx, Cy = point
        
        # take the min_y_pos/min_y_neg for this specific point
        my_pos = min_y_pos[i]
        my_neg = min_y_neg[i]
    
  
    
        # ----- Choose which y we use for POSITIVE direction -----
        if abs(my_pos) < tol2:
            y_plane = y - my_pos * factor
        elif abs(my_neg) < tol2:
            y_plane = y - my_neg * factor
        else:
            y_plane = y
    
        # -----------------------------------------------------------------
        # Now the plane masks are direction-dependent
        # -----------------------------------------------------------------
    
        # POSITIVE direction plane mask
        mask_plane = np.abs(boundary_xyz[:,1] - y_plane) < tol
        boundary_plane = boundary_xyz[mask_plane]
    
    
        # -----------------------------------------------------------------
        # POSITIVE x direction computation
        # -----------------------------------------------------------------
    
        if boundary_plane.size == 0:
            R2x_xpos.append(np.nan); R2z_xpos.append(np.nan)
        else:
            boundary_xz = boundary_plane[:, [0, 2]]
            deltas = boundary_xz - np.array([x, z])
            dx = deltas[:, 0]
    
            mask_pos = dx > 0
            if np.any(mask_pos):
                dp = deltas[mask_pos]
                d2 = np.sum(dp**2, axis=1)
                idx = np.argmin(d2)
                R2x_xpos.append(dp[idx,0])
                R2z_xpos.append(dp[idx,1])
            else:
                R2x_xpos.append(np.nan); R2z_xpos.append(np.nan)
    
        # -----------------------------------------------------------------
        # NEGATIVE x direction computation
        # -----------------------------------------------------------------
       
            mask_neg = dx < 0
            if np.any(mask_neg):
                dn = deltas[mask_neg]
                d2 = np.sum(dn**2, axis=1)
                idx = np.argmin(d2)
                R2x_xneg.append(dn[idx,0])
                R2z_xneg.append(dn[idx,1])
            else:
                R2x_xneg.append(np.nan); R2z_xneg.append(np.nan)
    
    
    
    #Centroid instead of IP for border X
    element_ids = COORDS_AM[:, 0].astype(int)   # shape (N,)
    unique_ids = np.unique(element_ids)
    cborder_xpos = np.zeros(len(element_ids), dtype=float)
    cborder_xneg = np.zeros(len(element_ids), dtype=float)
    centroid_xpos_arr=min_x_pos + IP_x
    centroid_xneg_arr=min_x_neg + IP_x
    
    for eid in unique_ids:
        idx = np.where(element_ids == eid)[0]      # rows with this ID
        centroid_val_xpos = np.max(centroid_xpos_arr[idx])
        centroid_val_xneg = np.min(centroid_xneg_arr[idx])
        cborder_xpos[idx] = centroid_val_xpos               # assign to all rows
        cborder_xneg[idx] = centroid_val_xneg               # assign to all rows
    
    
    
        
    
    
    # Compute AM sections
    min_x_pos_AM = cborder_xpos
    min_x_neg_AM = cborder_xneg
    min_y_pos_AM = min_y_pos
    min_y_neg_AM = min_y_neg
    min_z_pos_AM = min_z_pos
    min_z_neg_AM = min_z_neg
    Rx_AM = Rx
    Ry_AM = Ry
    Rz_AM = Rz
    R2x_xpos_AM=R2x_xpos
    R2x_xneg_AM=R2x_xneg
    R2z_xpos_AM=R2z_xpos
    R2z_xneg_AM=R2z_xneg
    IP_x_AM = IP_x
    IP_y_AM = IP_y
    IP_z_AM = IP_z
    overhang_gradient_AM = overhang_gradient
    
    # Compute BP setions
    Rx_BP = []
    Ry_BP = []
    Rz_BP = []
    
    coords_x = COORDS_BP[:, 1]
    coords_y = COORDS_BP[:, 2]
    coords_z = COORDS_BP[:, 3]
    coords_xyz = COORDS_BP[:, 1:4]
    
    x_min = np.min(boundary_xyz[:, 0])
    x_max = np.max(boundary_xyz[:, 0])
    y_min = np.min(boundary_xyz[:, 1])
    y_max = np.max(boundary_xyz[:, 1])
    z_min = np.min(COORDS_BP[:, 3])
    z_max = np.max(boundary_xyz[:, 2])  #boundary_xyz[:, 2] is z
    
    
    min_x_pos_BP = coords_x - x_min
    min_x_neg_BP = coords_x - x_max
    min_y_pos_BP = coords_y - y_min
    min_y_neg_BP = coords_y - y_max
    min_z_pos_BP = coords_z - z_min
    min_z_neg_BP = coords_z - z_max
    
    min_Cx_pos_BP = COORDS_BP[:, 4] - x_min
    min_Cx_neg_BP = COORDS_BP[:, 4] - x_max
    min_Cy_pos_BP = COORDS_BP[:, 5] - y_min
    min_Cy_neg_BP = COORDS_BP[:, 5] - y_max
    
    # Compute smallest absolute distances (pos and neg direction) for IP coordinates
    min_x_point = np.minimum(np.abs(min_x_pos_BP), np.abs(min_x_neg_BP))
    min_y_point = np.minimum(np.abs(min_y_pos_BP), np.abs(min_y_neg_BP))
    
    # Compute smallest absolute distances (pos and neg direction) for centroids
    min_x_centroid = np.minimum(np.abs(min_Cx_pos_BP), np.abs(min_Cx_neg_BP))
    min_y_centroid = np.minimum(np.abs(min_Cy_pos_BP), np.abs(min_Cy_neg_BP))
    
    # Compute IP as the difference
    IP_x_BP = min_x_point - min_x_centroid
    IP_y_BP = min_y_point - min_y_centroid
    IP_z_BP = COORDS_BP[:,3] - COORDS_BP[:,6]
    
    overhang_gradient_BP = np.zeros_like(coords_z)
    
    for point in coords_xyz:
        x, y, z = point
        
        # --- True Euclidean radial distance to boundary ---
        if boundary_xyz.size > 0:
            # Compute the vector differences from the current point to all boundary points
            deltas = boundary_xyz - np.array([x, y, z])  # Shape (N_boundary, 3)
        
            # Compute squared Euclidean distances
            squared_dists = np.sum(deltas**2, axis=1)
        
            # Get index of the closest boundary point
            min_idx = np.argmin(squared_dists)
            dx, dy, dz = deltas[min_idx]  # Corresponding dx, dy, dz
        
            # Append components of radius vector
            Rx_BP.append(dx)
            Ry_BP.append(dy)
            Rz_BP.append(dz)
        else:
            Rx_BP.append(np.nan)
            Ry_BP.append(np.nan)
            Rz_BP.append(np.nan)

    #2D R for BP points
    R2x_xpos_BP=min_x_pos_BP
    R2x_xneg_BP=min_x_neg_BP
    R2z_xpos_BP=min_z_pos_BP
    R2z_xneg_BP=min_z_neg_BP
    
    
    
    
    #Compute the points underneath
    
    # Full stack
    COORDS_stack = np.vstack([COORDS_AM, COORDS_BP])
    centroids_stack = COORDS_stack[:, 4:7]
    
    
    # Split indices
    n_AM = COORDS_AM.shape[0]     # number of AM points
    n_BP = COORDS_BP.shape[0]     # number of BP points
    
    AM_centroids = centroids_stack[:n_AM]     # only AM part
    BP_centroids = centroids_stack[n_AM:]     # only BP part (not needed for loop)
    
    step_size = 0.6
    tol = 0.1
    z_min = np.min(COORDS_AM[:, 3])
    underneath_AM = COORDS_AM[:,3]   # we compute these
    underneath_BP = np.full(n_BP, 0)
    
    
    # Compute walls###################################################
    # Extract X, Y, Z
    boundary_xyz = boundary[:, 1:4]  # X, Y, Z
    b_x = boundary_xyz[:, 0]
    b_y = boundary_xyz[:, 1]
    b_z = boundary_xyz[:, 2]
    
    # Filter points by Y range
    mask = (b_y >= -0.4) & (b_y <= 0.4)
    
    # Select points within the mask
    walls_x = b_x[mask]
    walls_y = b_y[mask]
    walls_z = b_z[mask]
    
    # Round coordinates to bin by 0.1 for grouping
    walls_x_rounded = np.round(walls_x / 0.1) * 0.1
    
    # Unique x values
    walls_x_unique = np.unique(walls_x_rounded)
    
    # Parameters
    tol_x = 0.1      # tolerance in x for grouping
    z_neigh_dist = 0.6  # required distance in z to consider neighbors
    z_tol = 0.1     # small tolerance for numerical stability
    
    walls = []  # will store [wall_x, min_z]
    

    for wx in walls_x_unique:
        # Points within x-tolerance
        group_mask = np.abs(walls_x_rounded - wx) <= tol_x
        z_vals = walls_z[group_mask]
    
        # Sort z-values ascending
        z_vals = np.sort(z_vals)
        
        # Track points that are part of valid neighbor pairs
        neighbor_points = []
        
        for i in range(len(z_vals)):
            for j in range(i+1, len(z_vals)):
                if abs(z_vals[j] - z_vals[i] - z_neigh_dist) <= z_tol:
                    neighbor_points.extend([z_vals[i], z_vals[j]])
    
        if neighbor_points:
            # Compute min_z among only the neighbor points
            min_z = np.min(neighbor_points)
            walls.append([wx, min_z])
    
    walls = np.array(walls)
    
    # Pre-allocate arrays for distances
    dist_pos_x = []  # distance to nearest wall in positive X
    dist_neg_x = []  # distance to nearest wall in negative X
    
    
    coords_xyz = COORDS_AM[:, 1:4]
    
    for point in coords_xyz:
        x, y, z = point
        
        # Filter walls that are above current z
        valid_walls = walls[walls[:, 1] < z]
        
        if len(valid_walls) == 0:
            # No wall above point
            dist_pos_x.append(np.nan)
            dist_neg_x.append(np.nan)
            print("no valid walls for this point")
            print("walls")
            print(walls)
            print("z")
            print(z)
            continue
        
        # Compute distances along X
        dx = valid_walls[:, 0] - x  # difference in X
        
        # Positive X distance: wall_x > x
        pos_dist = dx[dx > 0]
        dist_pos_x.append(np.min(pos_dist) if len(pos_dist) > 0 else np.nan)
        
        # Negative X distance: wall_x < x
        neg_dist = dx[dx < 0]  # keep distances negative
        dist_neg_x.append(np.max(neg_dist) if len(neg_dist) > 0 else np.nan)
    
    # Convert to arrays
    walls_xpos_AM = dist_pos_x
    walls_xneg_AM = dist_neg_x
    
    
    
    walls_xpos_BP = min_x_pos_BP
    walls_xneg_BP = min_x_neg_BP
            
            
        
    
    
    #STACK AM and BP sections
    
    min_x_pos = np.vstack([
    np.array(min_x_pos_AM).reshape(-1, 1),
    np.array(min_x_pos_BP).reshape(-1, 1)    ])
    
    min_x_neg = np.vstack([
        np.array(min_x_neg_AM).reshape(-1, 1),
        np.array(min_x_neg_BP).reshape(-1, 1)    ])
    
    min_y_pos = np.vstack([
        np.array(min_y_pos_AM).reshape(-1, 1),
        np.array(min_y_pos_BP).reshape(-1, 1)    ])
    
    min_y_neg = np.vstack([
        np.array(min_y_neg_AM).reshape(-1, 1),
        np.array(min_y_neg_BP).reshape(-1, 1)    ])
    
    min_z_pos = np.vstack([
        np.array(min_z_pos_AM).reshape(-1, 1),
        np.array(min_z_pos_BP).reshape(-1, 1)    ])
    
    min_z_neg = np.vstack([
        np.array(min_z_neg_AM).reshape(-1, 1),
        np.array(min_z_neg_BP).reshape(-1, 1)    ])
    
    R2x_xpos = np.vstack([
        np.array(R2x_xpos_AM).reshape(-1, 1),
        np.array(R2x_xpos_BP).reshape(-1, 1)    ])
    
    R2x_xneg = np.vstack([
        np.array(R2x_xneg_AM).reshape(-1, 1),
        np.array(R2x_xneg_BP).reshape(-1, 1)    ])
    
    R2z_xpos = np.vstack([
        np.array(R2z_xpos_AM).reshape(-1, 1),
        np.array(R2z_xpos_BP).reshape(-1, 1)    ])
    
    R2z_xneg = np.vstack([
        np.array(R2z_xneg_AM).reshape(-1, 1),
        np.array(R2z_xneg_BP).reshape(-1, 1)    ])
    
    
    Rx = np.vstack([
        np.array(Rx_AM).reshape(-1, 1),
        np.array(Rx_BP).reshape(-1, 1)    ])
    
    Ry = np.vstack([
        np.array(Ry_AM).reshape(-1, 1),
        np.array(Ry_BP).reshape(-1, 1)    ])
    
    Rz = np.vstack([
        np.array(Rz_AM).reshape(-1, 1),
        np.array(Rz_BP).reshape(-1, 1)    ])
    
    
    IP_x = np.vstack([
        np.array(IP_x_AM).reshape(-1, 1),
        np.array(IP_x_BP).reshape(-1, 1)    ])
    
    IP_y = np.vstack([
        np.array(IP_y_AM).reshape(-1, 1),
        np.array(IP_y_BP).reshape(-1, 1)    ])
    
    IP_z = np.vstack([
        np.array(IP_z_AM).reshape(-1, 1),
        np.array(IP_z_BP).reshape(-1, 1)    ])
    
    
    
    walls_xpos = np.vstack([
        np.array(walls_xpos_AM).reshape(-1, 1),
        np.array(walls_xpos_BP).reshape(-1, 1)    ])
    
    
    walls_xneg = np.vstack([
        np.array(walls_xneg_AM).reshape(-1, 1),
        np.array(walls_xneg_BP).reshape(-1, 1)    ])
    
    
    underneath = np.vstack([
        np.array(underneath_AM).reshape(-1, 1),
        np.array(underneath_BP).reshape(-1, 1)    ])
    
    overhang_gradient = np.vstack([overhang_gradient_AM.reshape(-1, 1), overhang_gradient_BP.reshape(-1, 1)])
    
    COORDS = np.vstack([COORDS_AM, COORDS_BP])
    
    Eigenstrains = np.vstack([Eigenstrains_AM, Eigenstrains_BP])
    
    
    
    # Restore original order
    original_order = np.argsort(np.concatenate([idx_AM, idx_BP]))
    
    # Reorder all stacked arrays
    min_x_pos = min_x_pos[original_order]
    min_x_neg = min_x_neg[original_order]
    min_y_pos = min_y_pos[original_order]
    min_y_neg = min_y_neg[original_order]
    min_z_pos = min_z_pos[original_order]
    min_z_neg = min_z_neg[original_order]
    
    Rx = Rx[original_order]
    Ry = Ry[original_order]
    Rz = Rz[original_order]
    
    R2x_xpos = R2x_xpos[original_order]
    R2z_xpos = R2z_xpos[original_order]
    R2x_xneg = R2x_xneg[original_order]
    R2z_xneg = R2z_xneg[original_order]
    
    
    walls_xpos = walls_xpos[original_order]
    walls_xneg = walls_xneg[original_order]
    
    underneath = underneath[original_order]
    
    IP_x = IP_x[original_order]
    IP_y = IP_y[original_order]
    IP_z = IP_z[original_order]
    
    COORDS = COORDS[original_order]
    Eigenstrains = Eigenstrains[original_order]
    overhang_gradient = overhang_gradient[original_order]
    
    
    
    
    # #Translation
    # COORDS[:, 1] += X_trans  # X
    # COORDS[:, 2] += Y_trans  # Y
    # COORDS[:, 3] += Z_trans  # Z
    # COORDS[:, 4] += X_trans  # X
    # COORDS[:, 5] += Y_trans  # Y
    # COORDS[:, 6] += Z_trans  # Z
    
    
    # boundary[:, 1] += X_trans  # X
    # boundary[:, 2] += Y_trans  # Y
    # boundary[:, 3] += Z_trans  # Z
    
    
    # Compose node_ip
    node_ip = np.hstack((
        COORDS[:, 0:4],  # ElementID, X, Y, Z; ElementID will be removed later during copying
        np.array(min_x_pos),
        np.array(min_x_neg),
        np.array(min_y_pos),
        np.array(min_y_neg),
        np.array(min_z_pos),
        np.array(min_z_neg),
        np.array(R2x_xpos),
        np.array(R2z_xpos),
        np.array(R2x_xneg),
        np.array(R2z_xneg),
        np.array(IP_x),
        np.array(IP_y),
        np.array(IP_z),
        np.array(walls_xpos),
        np.array(walls_xneg),
        np.array(underneath),
        overhang_gradient    ))
    
    node_op=Eigenstrains[:,0:6]
    border_nodes=boundary[:, 1:4]
    # border_nodes=overhang[:, 1:4]

    
    return node_ip, node_op, border_nodes

def process_single_file(file_info, dataset_dir):
    csv_file = file_info["filename"]
    boundary_file = file_info["boundary_file"]
    Rot = file_info["Rot"]
    X_trans = file_info["X_trans"]
    Y_trans = file_info["Y_trans"]
    Z_trans = file_info["Z_trans"]

    file_path = os.path.join(dataset_dir, csv_file)
    file_path_boundary = os.path.join(dataset_dir, boundary_file)

    print(f"Processing: {csv_file}, Rot={Rot}, X_trans={X_trans}, Y_trans={Y_trans}, Z_trans={Z_trans}, {boundary_file}")

    # Load boundary data
    boundary_raw = pd.read_csv(file_path_boundary, sep=',', usecols=[0, 1, 2, 3, 4], header=None)
    boundary_raw.columns = ['NodeID', 'X', 'Y', 'Z', 'boundaryflag']
    

    # Load main data
    raw_data = pd.read_csv(file_path, sep=',', usecols=[4, 6, 7, 8, 12, 13, 14, 15, 16, 17], skiprows=0)
    raw_data.columns = ['ElementID', 'X', 'Y', 'Z', 'ES11', 'ES22', 'ES33', 'ES12', 'ES23', 'ES13']
    COORDS = raw_data[['ElementID', 'X', 'Y', 'Z']]
    Eigenstrains = raw_data[['ES11', 'ES22', 'ES33', 'ES12', 'ES23', 'ES13']]

    # Compute centroids
    centroids = COORDS.groupby('ElementID')[['X', 'Y', 'Z']].mean().reset_index()
    centroids.columns = ['ElementID', 'Centroid_X', 'Centroid_Y', 'Centroid_Z']

    # Merge coords with centroids
    COORDS_with_centroids = pd.merge(COORDS, centroids, on='ElementID', how='left')
    COORDS_with_centroids = COORDS_with_centroids[['ElementID', 'X', 'Y', 'Z', 'Centroid_X', 'Centroid_Y', 'Centroid_Z']].to_numpy()
    Eigenstrains = Eigenstrains.to_numpy()


    node_ip, node_op, border_points = data_prep(COORDS_with_centroids, Eigenstrains, boundary_raw, Rot, X_trans, Y_trans, Z_trans)

    # Return dict instead of tuple for easier unpacking later
    return {
        'vertices': node_ip,
        'nodal_eigenstrain': node_op,
        'points_cloud': border_points,
        'file_name': csv_file
    }

# === MAIN LOOP ===
num_workers = 63

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(process_single_file, csv_files_training, itertools.repeat(dataset_dir)))

# # Sequential loop
# results = []
# for file in csv_files_training:
#     result = process_single_file(file, dataset_dir)
#     results.append(result)

# Reorganize the results into all_data_original dictionary
all_data_training = {
    'vertices': [res['vertices'] for res in results],
    'nodal_eigenstrain': [res['nodal_eigenstrain'] for res in results],
    'points_cloud': [res['points_cloud'] for res in results],
    'file_name': [res['file_name'] for res in results],
    'w': [],
    'L': []
}

# Number of times to replicate all_data_original
num_iterations = 6  # Set >=8 in actual use case for effective data reduction

all_data = {
    'vertices': [],
    'nodal_eigenstrain': [],
    'points_cloud': [],
    'file_name': []
}

# Exclude the last entry during the loop
for iter_idx in range(num_iterations):
    for i, file in enumerate(csv_files_training):  # Skip last entry
        X_trans, Y_trans, Z_trans = 0, 0, 0

        vertices = copy.deepcopy(all_data_training['vertices'][i])  # shape (N, >=13 columns)
        nodal_eigenstrain = copy.deepcopy(all_data_training['nodal_eigenstrain'][i])
        points_cloud = copy.deepcopy(all_data_training['points_cloud'][i])
        file_name = all_data_training['file_name'][i]

        N = vertices.shape[0]

        # Calculate radius from dx, dy, dz columns (indices 10:13)
        deltas = vertices[:, 10:12]  # shape (N, 3)
        radius_pos = np.linalg.norm(deltas, axis=1)
        
        deltas = vertices[:, 12:14]  # shape (N, 3)
        radius_neg = np.linalg.norm(deltas, axis=1)

        # Condition for columns 9 and 10 (indices 8 and 9) absolute values < 1.8
        cond_cols_9_10 = (np.abs(vertices[:, 8]) < 1.2) & (np.abs(vertices[:, 9]) < 1.2)
        
        # Condition for columns 8 and 9 (indices 6 and 7) absolute values < 1.8
        cond_cols_7_8 = (np.abs(vertices[:, 6]) < 1.2) & (np.abs(vertices[:, 7]) < 1.2)

        # Combined condition: radius < 1.8 AND columns 9 & 10 < 1.8
        mask_important = (radius_pos < 0.6) | (radius_neg < 0.6) 

        # For points important, randomly keep ~50%
        keep_fraction = 0.25
        important_indices = np.where(mask_important)[0]
        num_important_to_keep = int(len(important_indices) * keep_fraction)
        keep_important_indices = np.random.choice(
            important_indices,
            size=num_important_to_keep,
            replace=False
        ) if num_important_to_keep > 0 else np.array([], dtype=int)
        


        # For points NOT important, randomly keep ~20%
        mask_non_important = ~mask_important
        non_important_indices = np.where(mask_non_important)[0]

        keep_fraction = 0.03
        num_non_important_to_keep = int(len(non_important_indices) * keep_fraction)
        keep_non_important_indices = np.random.choice(
            non_important_indices,
            size=num_non_important_to_keep,
            replace=False
        ) if num_non_important_to_keep > 0 else np.array([], dtype=int)

        # Combine indices: all important + selected non-important
        keep_indices = np.concatenate([keep_important_indices, keep_non_important_indices])
        

        # Subset vertices, nodal_eigenstrain based on keep_indices
        vertices = vertices[keep_indices]
        nodal_eigenstrain = nodal_eigenstrain[keep_indices]

        # Sort by X coordinate which is column 1 (since 0 is ID)
        sort_idx = np.argsort(vertices[:, 1])
        vertices = vertices[sort_idx]
        nodal_eigenstrain = nodal_eigenstrain[sort_idx]

        # Remove ID column (column 0)
        vertices_no_id = vertices[:, 1:]
        vertices_no_id[:, :3] += [X_trans, Y_trans, Z_trans]
        points_cloud[:, :3] += [X_trans, Y_trans, Z_trans]

        all_data['vertices'].append(vertices_no_id)
        all_data['nodal_eigenstrain'].append(nodal_eigenstrain)
        all_data['points_cloud'].append(points_cloud)
        all_data['file_name'].append(file_name)
        
# === Process all validation entries ===
for val_file_info in csv_files_validation:
    val_data = process_single_file(val_file_info, dataset_dir)
    X_trans, Y_trans, Z_trans = 0, 0, 0

    vertices = copy.deepcopy(val_data['vertices'])
    nodal_eigenstrain = copy.deepcopy(val_data['nodal_eigenstrain'])
    points_cloud = copy.deepcopy(val_data['points_cloud'])
    file_name = val_data['file_name']

    N = vertices.shape[0]

    # Calculate radius from dx, dy, dz columns (indices 10:13)
    deltas = vertices[:, 10:12]  # shape (N, 3)
    radius_pos = np.linalg.norm(deltas, axis=1)
    
    deltas = vertices[:, 12:14]  # shape (N, 3)
    radius_neg = np.linalg.norm(deltas, axis=1)

    # Condition for columns 9 and 10 (indices 8 and 9) absolute values < 1.8
    cond_cols_9_10 = (np.abs(vertices[:, 8]) < 1.2) & (np.abs(vertices[:, 9]) < 1.2)
    
    # Condition for columns 8 and 9 (indices 6 and 7) absolute values < 1.8
    cond_cols_7_8 = (np.abs(vertices[:, 6]) < 1.2) & (np.abs(vertices[:, 7]) < 1.2)

    # Combined condition: radius < 1.8 AND columns 9 & 10 < 1.8
    mask_important = (radius_pos < 0.6) | (radius_neg < 0.6) 

    # For points important, randomly keep ~50%
    keep_fraction = 0.25
    important_indices = np.where(mask_important)[0]
    num_important_to_keep = int(len(important_indices) * keep_fraction)
    keep_important_indices = np.random.choice(
        important_indices,
        size=num_important_to_keep,
        replace=False
    ) if num_important_to_keep > 0 else np.array([], dtype=int)
    


    # For points NOT important, randomly keep ~20%
    mask_non_important = ~mask_important
    non_important_indices = np.where(mask_non_important)[0]

    keep_fraction = 0.03
    num_non_important_to_keep = int(len(non_important_indices) * keep_fraction)
    keep_non_important_indices = np.random.choice(
        non_important_indices,
        size=num_non_important_to_keep,
        replace=False
    ) if num_non_important_to_keep > 0 else np.array([], dtype=int)

    # Combine indices: all important + selected non-important
    keep_indices = np.concatenate([keep_important_indices, keep_non_important_indices])
    

    # Subset vertices, nodal_eigenstrain based on keep_indices
    vertices = vertices[keep_indices]
    nodal_eigenstrain = nodal_eigenstrain[keep_indices]

    # Sort by X coordinate which is column 1 (since 0 is ID)
    sort_idx = np.argsort(vertices[:, 1])
    vertices = vertices[sort_idx]
    nodal_eigenstrain = nodal_eigenstrain[sort_idx]

    # Remove ID column (column 0)
    vertices_no_id = vertices[:, 1:]
    vertices_no_id[:, :3] += [X_trans, Y_trans, Z_trans]
    points_cloud[:, :3] += [X_trans, Y_trans, Z_trans]

    all_data['vertices'].append(vertices_no_id)
    all_data['nodal_eigenstrain'].append(nodal_eigenstrain)
    all_data['points_cloud'].append(points_cloud)
    all_data['file_name'].append(file_name)

# === Process all test entries ===
for test_file_info in csv_files_test:
    test_data = process_single_file(test_file_info, dataset_dir)

    X_trans, Y_trans, Z_trans = 0, 0, 0  # Optional: set random shift if needed

    vertices = copy.deepcopy(test_data['vertices'])
    nodal_eigenstrain = copy.deepcopy(test_data['nodal_eigenstrain'])
    points_cloud = copy.deepcopy(test_data['points_cloud'])
    file_name = test_data['file_name']

    # === Optional subsample or sort
    # N = vertices.shape[0]
    # keep_size = int(N / 8)
    # indices = np.random.choice(N, size=keep_size, replace=False)
    # vertices = vertices[indices]
    # nodal_eigenstrain = nodal_eigenstrain[indices]
    # sort_idx = np.argsort(vertices[:, 0])
    # vertices = vertices[sort_idx]
    # nodal_eigenstrain = nodal_eigenstrain[sort_idx]

    # === Transform and append
    vertices_no_id = vertices[:, 1:]
    vertices_no_id[:, :3] += [X_trans, Y_trans, Z_trans]
    points_cloud[:, :3] += [X_trans, Y_trans, Z_trans]

    all_data['vertices'].append(vertices_no_id)
    all_data['nodal_eigenstrain'].append(nodal_eigenstrain)
    all_data['points_cloud'].append(points_cloud)
    all_data['file_name'].append(file_name)

# === DONE ===
print(f"\nProcessed {len(csv_files_training)} training datasets + {len(csv_files_validation)} validation datasets + {len(csv_files_test)} test dataset.")

# === SAVE TO PICKLE ===
dataset_dir = "data/Eigenstrain_data"
output_path = os.path.join(dataset_dir, "Eigenstrain_DatasetILL_FI_BorderR_AM_loopN001-120_Val_ILLG_CL48_IP_OHBZ3E_Rred18-005_AMBP_0-16_Rot_Rzx.pkl")
with open(output_path, "wb") as f:
    pickle.dump(all_data, f)

print(f"Saved all_data to: {output_path}")


# Step: Visualize point cloud from all_data['points_cloud'][0]
points_cloud = all_data['points_cloud'][0]  # shape (N, 3)

# Create Open3D point cloud object
pcd_cloud = o3d.geometry.PointCloud()
pcd_cloud.points = o3d.utility.Vector3dVector(points_cloud)

# Assign default color (e.g., gray)
colors = np.tile([0.3, 0.3, 0.3], (points_cloud.shape[0], 1))
pcd_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd_cloud])


















