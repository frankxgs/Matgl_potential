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
from matgl.graph.converters import GraphConverter
from matgl.ext.pymatgen import Molecule2Graph, get_element_list
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.layers import BondExpansion
from matgl.models import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule


class NewMEGNet(MEGNet):
    def predict_structure(
            self,
            structure,
            state_attr: torch.Tensor | None = None,
            graph_converter: GraphConverter | None = None
    ):
        g, lat, state_attr_default = graph_converter.get_graph(structure)
        # Move tensors to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = g.to(device)
        lat = lat.to(device)
        state_attr_default = torch.tensor(state_attr_default).to(device)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        if state_attr is None:
            state_attr = torch.tensor(state_attr_default)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["edge_attr"] = self.bond_expansion(bond_dist)
        return self(g=g, state_attr=state_attr).detach()


cache_1 = torch.load('data/potentials.pt')
cache_2 = torch.load('data/rp.pt')
non_nan_idx = (~torch.isnan(cache_1['labels']['RP'])).nonzero().squeeze()
ids = cache_2['ids']
Molecules = cache_2['Molecules']
Mols = []
for i in non_nan_idx:
    Mols.append(cache_1["Molecules"][i])
ops = cache_1['labels']['RP'][non_nan_idx].tolist()
scaler = MinMaxScaler(feature_range=(0,1))
ops = np.array(ops).reshape(-1, 1)
ops = scaler.fit_transform(ops).flatten().tolist()

elem_list = get_element_list(Mols)
converter = Molecule2Graph(element_types=elem_list, cutoff=4.0)
node_embed = torch.nn.Embedding(len(elem_list), 16)
bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)
model = NewMEGNet(
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
best_model_path = 'checkpoints/RP/version_0/model-epoch=232-val_Total_Loss=0.01.ckpt'
checkpoint = torch.load(best_model_path)
state_dict = checkpoint['state_dict']
state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items() if k.startswith('model.')}
model.load_state_dict(state_dict)
model = model.to('cuda')
model.eval()
rp_predictions = []
with torch.no_grad():
    for mol in tqdm(Molecules):
        op_pred = model.predict_structure(structure=mol, graph_converter=converter)
        rp_predictions.append(op_pred.to('cpu'))

rp_scaled = np.array(rp_predictions).reshape(-1, 1)
rp_scaled = scaler.inverse_transform(rp_scaled)
rp_predictions = rp_scaled.flatten().tolist()

predictions = {'id':ids, 'reduction_potential':rp_predictions}
df_pred = pd.DataFrame(predictions)
df_pred.to_csv('rp_preds_1.csv', index=False)