import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append('nnue-pytorch')
from model.config import NNUELightningConfig
from model.lightning_module import NNUE

def eval_loop(model, val_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = [t.to(model.device) for t in batch]
            loss = model.step_(batch, 0, "val_loss")
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = [t.to(model.device) for t in batch]
        optimizer.zero_grad()
        loss = model.step_(batch, 0, "train_loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== ESTUDIO DE ABLACIÓN: CONGELANDO BASELINE ===")
    
    config = NNUELightningConfig()
    config.loss_params.start_lambda = 1.0
    config.loss_params.end_lambda = 1.0
    model = NNUE(config=config, max_epoch=1, num_batches_per_epoch=1).to(device)
    
    print("[1] Cargando pesos del Baseline (best_model_baseline.pt)...")
    model.load_state_dict(torch.load("best_model_baseline.pt", map_location=device))
    
    print("[2] Congelando arquitectura HalfKPA (Freezing)...")
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    print("[3] Activando aprendizaje SÓLO en W_e...")
    model.model.W_e.weight.requires_grad = True
    
    print("[4] Cargando dataset de 200k...")
    data = torch.load("mvp_200k_dataset.pt")
    dataset = TensorDataset(
        data['us'], data['them'], data['white_indices'], data['white_values'],
        data['black_indices'], data['black_values'], data['outcome'],
        data['score'], data['psqt_indices'], data['layer_stack_indices'],
        data['causal_embeddings']
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    best_val_loss = float('inf')
    
    print("\nIniciando entrenamiento...")
    for epoch in range(100):
        t_loss = train_epoch(model, train_loader, optimizer)
        v_loss = eval_loop(model, val_loader)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "best_model_nnuecc.pt") # OVERWRITE NNUECC!
            marker = "<- BEST"
        else:
            marker = ""
            
        print(f"Epoch {epoch+1:03d}/100 | Train: {t_loss:.5f} | Val: {v_loss:.5f} {marker}")
        
    print("\n=== ENTRENAMIENTO FINALIZADO ===")
    print("El modelo best_model_nnuecc.pt ahora es una versión ESTRICTAMENTE mejorada del Baseline.")

if __name__ == "__main__":
    main()
