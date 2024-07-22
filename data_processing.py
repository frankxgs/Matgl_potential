from __future__ import annotations

import os
import logging
import pandas as pd
import torch
from pymatgen.core import Molecule
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


data_df = pd.read_csv('data/potential.csv')

logging.basicConfig(filename='molecule_processing.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
chache_path = 'data/potentials.pt'
if os.path.exists(chache_path):
    cache = torch.load(chache_path)
    ids =cache['ids']
    Molecules = cache["Molecules"]
    labels = cache['labels']
else:
    ids = []
    ops = []
    rps = []
    Molecules = []
    for i in tqdm(range(len(data_df))):
        smiles = data_df['smiles'][i]
        op = data_df['oxidation_potential'][i]
        rp = data_df['reduction_potential'][i]
        idx = data_df['index'][i]
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if rdkit_mol is None:
            logging.warning(f"Invalid SMILES '{smiles}' at index {idx}. Skipping...")
            continue
        rdkit_mol = Chem.AddHs(rdkit_mol)
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                if AllChem.EmbedMolecule(rdkit_mol) == -1:
                    continue
                if AllChem.UFFOptimizeMolecule(rdkit_mol) == 0:
                    break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: Optimization failed with error {e}")
        if rdkit_mol.GetNumConformers() == 0:
            logging.warning(f"Molecule {smiles} has 0 conformers. Skipping...")
            continue
        atoms = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
        conformer = rdkit_mol.GetConformer()
        coordinates = conformer.GetPositions()
        pmg_mol = Molecule(species=atoms, coords=coordinates)
        Molecules.append(pmg_mol)
        ops.append(op)
        rps.append(rp)
        ids.append(idx)
    labels = {'OP': torch.tensor(ops), 'RP': torch.tensor(rps)}
    torch.save({'ids': ids, 'Molecules': Molecules, 'labels': labels}, f=chache_path)


