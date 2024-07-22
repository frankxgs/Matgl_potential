import zipfile
import logging
import os
os.environ['DGLBACKEND'] = 'pytorch' 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightning as pl
import torch

from dgl.data.utils import split_dataset
from pymatgen.core import Molecule
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matgl.ext.pymatgen import Molecule2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.layers import BondExpansion
from matgl.models import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule


cache = torch.load('data/potentials.pt')
non_nan_idx = (~torch.isnan(cache['labels']['OP'])).nonzero().squeeze()
Molecules = []
for i in non_nan_idx:
    Molecules.append(cache["Molecules"][i])
ops = cache['labels']['OP'][non_nan_idx].tolist()
scaler = MinMaxScaler(feature_range=(0,1))
ops = np.array(ops).reshape(-1, 1)
ops = scaler.fit_transform(ops).flatten().tolist()
labels = {'OP': ops}

elem_list = get_element_list(Molecules)
converter = Molecule2Graph(element_types=elem_list, cutoff=4.0)
dataset = MGLDataset(
    save_dir='MGLDataset/OP',
    structures=Molecules,
    labels=labels,
    converter=converter
)

train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)


train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_graph,
    batch_size=512,
    num_workers=0,
)
print("Test Dataset:", len(test_data))

node_embed = torch.nn.Embedding(len(elem_list), 16)
bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)
model = MEGNet(
    dim_node_embedding=16,
    dim_edge_embedding=100,
    dim_state_embedding=2,
    nblocks=3,
    hidden_layer_sizes_input=(64, 32),
    hidden_layer_sizes_conv=(64, 64, 32),
    nlayers_set2set=1,
    niters_set2set=2,
    hidden_layer_sizes_output=(32, 16),
    is_classification=False,
    activation_type="softplus2",
    bond_expansion=bond_expansion,
    cutoff=4.0,
    gauss_width=0.5,
)
best_model_path = 'checkpoints/OP/version_5/model-epoch=728-val_Total_Loss=0.00.ckpt'
lit_model = ModelLightningModule.load_from_checkpoint(model=model, checkpoint_path=best_model_path)
trainer = pl.Trainer(accelerator='gpu')
trainer.test(model=lit_model, dataloaders=test_loader)
