import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

import sys
sys.path.append('nnue-pytorch')
from model.config import NNUELightningConfig
from model.lightning_module import NNUE
from model.dataset import NNUEDataset

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        us, them, white_indices, white_values, black_indices, black_values, psqt_indices, psqt_values, score, outcome, planes = batch
        
        us = us.to('cuda')
        them = them.to('cuda')
        white_indices = white_indices.to('cuda')
        white_values = white_values.to('cuda')
        black_indices = black_indices.to('cuda')
        black_values = black_values.to('cuda')
        psqt_indices = psqt_indices.to('cuda')
        psqt_values = psqt_values.to('cuda')
        score = score.to('cuda')
        outcome = outcome.to('cuda')
        planes = planes.to('cuda')
        
        batch_size = us.size(0)
        
        with torch.no_grad():
            x = planes.view(batch_size, -1)
            causal_embeddings = cnn.get_embedding(x)
            
        optimizer.zero_grad()
        
        preds = model.model(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, psqt_values, causal_embeddings)
        preds = preds.squeeze(-1)
        
        loss = nn.MSELoss()(preds, score.squeeze(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def eval_loop(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            us, them, white_indices, white_values, black_indices, black_values, psqt_indices, psqt_values, score, outcome, planes = batch
            
            us = us.to('cuda')
            them = them.to('cuda')
            white_indices = white_indices.to('cuda')
            white_values = white_values.to('cuda')
            black_indices = black_indices.to('cuda')
            black_values = black_values.to('cuda')
            psqt_indices = psqt_indices.to('cuda')
            psqt_values = psqt_values.to('cuda')
            score = score.to('cuda')
            outcome = outcome.to('cuda')
            planes = planes.to('cuda')
            
            batch_size = us.size(0)
            x = planes.view(batch_size, -1)
            causal_embeddings = cnn.get_embedding(x)
            
            preds = model.model(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, psqt_values, causal_embeddings)
            preds = preds.squeeze(-1)
            
            loss = nn.MSELoss()(preds, score.squeeze(-1))
            total_loss += loss.item()
            
    return total_loss / len(loader)

if __name__ == "__main__":
    from prev_move_models import CNNNet
    
    print("=== INICIANDO ENTRENAMIENTO CONGELADO (FREEZING) ===")
    cnn = CNNNet(n_moves=1928).to('cuda')
    state = torch.load("../models/prev_move_cnn_v2.pt", map_location='cuda')
    cnn.load_state_dict(state['model'] if 'model' in state else state)
    cnn.eval()

    config = NNUELightningConfig()
    model = NNUE(config=config, max_epoch=1, num_batches_per_epoch=1).to('cuda')
    
    # 1. Load the Baseline (so we have the exact same base knowledge)
    print("Cargando best_model_baseline.pt...")
    model.load_state_dict(torch.load("best_model_baseline.pt", map_location='cuda'))
    
    # 2. FREEZE all layers
    print("Congelando arquitectura base...")
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    # 3. ONLY UNFREEZE W_e
    model.model.W_e.weight.requires_grad = True
    print("Descongelando SOLO la matriz W_e.")
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    print("Cargando datasets...")
    full_dataset = torch.load("mvp_200k_dataset.pt")
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    best_val = float('inf')
    
    for epoch in range(100):
        t_loss = train_epoch(model, train_loader, optimizer)
        v_loss = eval_loop(model, val_loader)
        
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), "best_model_nnuecc_frozen.pt")
            marker = "<- BEST"
        else:
            marker = ""
            
        print(f"Epoch {epoch+1:03d} | Train: {t_loss:.6f} | Val: {v_loss:.6f} {marker}")
