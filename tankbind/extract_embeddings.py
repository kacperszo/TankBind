import logging
import sys

import torch

sys.path.insert(0, "/work/tankbind")
from data import TankBind_prediction
from model import get_model
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO)
dataset = TankBind_prediction("/work/examples/single_PDB_6hd6/6hd6_dataset/")
loader = DataLoader(dataset, batch_size=5, follow_batch=["x", "y", "compound_pair"])

model = get_model(0, logging, "cpu")
model.load_state_dict(torch.load("/work/saved_models/self_dock.pt", map_location="cpu"))
model.eval()

# tranistion jest wywoływany 5× (jeden per stack) — nadpisujemy za każdym razem,
# po forward pass zostaje wynik ostatniego (finalnego) wywołania
captured = {}
model.tranistion.register_forward_hook(
    lambda m, inp, out: captured.__setitem__('z', out.detach())
)
# pre-hook: inp to tuple argumentów forward(z, z_mask), bierzemy z_mask
# z_mask nie zmienia się między stackami — wystarczy złapać raz z pierwszego modułu
model.triangle_self_attention_list[0].register_forward_pre_hook(
    lambda m, inp: captured.__setitem__('z_mask', inp[1].detach())
)

batch = next(iter(loader))
with torch.no_grad():
    y_pred, affinity_pred = model(batch)

z      = captured['z']       # [B, n_prot, n_comp, 128]
z_mask = captured['z_mask']  # [B, n_prot, n_comp]

# masked mean pool → jeden wektor 128-dim na kompleks
z_pooled = (z * z_mask.unsqueeze(-1)).sum(dim=(1, 2)) / \
           z_mask.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)  # [B, 128]

print("z shape:       ", z.shape)
print("z_pooled shape:", z_pooled.shape)

torch.save({'z_pooled': z_pooled, 'z': z}, '/work/outputs/embeddings.pt')
