import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import lightning as L
import argparse

sys.path.append('nnue-pytorch')
from model.config import NNUELightningConfig
from model.lightning_module import NNUE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "nnuecc"], required=True)
    parser.add_argument("--input", default="mvp_50k_dataset.pt")
    args = parser.parse_args()
    
    print(f"=== Iniciando Fine-Tuning: {args.mode.upper()} ===")
    
    print(f"[1] Cargando dataset {args.input}...")
    data = torch.load(args.input)
    
    # We use causal_embeddings always, but in baseline we just multiply by 0
    dataset = TensorDataset(
        data['us'], data['them'], data['white_indices'], data['white_values'],
        data['black_indices'], data['black_values'], data['outcome'],
        data['score'], data['psqt_indices'], data['layer_stack_indices'],
        data['causal_embeddings']
    )
    
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    print("[2] Configurando modelo NNUE...")
    config = NNUELightningConfig()
    config.optimizer_config.lr = 1e-4 # Standard fine-tuning LR
    config.loss_params.start_lambda = 1.0
    config.loss_params.end_lambda = 1.0
    
    model = NNUE(
        config=config,
        max_epoch=10,
        num_batches_per_epoch=len(dataloader),
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # In Baseline, we FREEZE W_e (it stays 0).
    # In NNUECC, we UNFREEZE W_e.
    # We UNFREEZE the rest of the network for both to allow fine-tuning.
    for name, param in model.named_parameters():
        if "W_e" in name:
            param.requires_grad = (args.mode == "nnuecc")
        else:
            param.requires_grad = True # Unfreeze base network for both

    print("\n[3] Iniciando Entrenamiento por 10 epochs...")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    model.train()
    history = []
    for epoch in range(10):
        total_loss = 0.0
        for batch in dataloader:
            batch = [t.to(model.device) for t in batch]
            optimizer.zero_grad()
            
            if args.mode == "baseline":
                # Mask causal embeddings to 0.0 to simulate no PMP
                batch[-1] = torch.zeros_like(batch[-1])
                
            loss = model.step_(batch, 0, "train_loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")
        history.append(avg_loss)
        
    print("\n=== Entrenamiento Finalizado ===")
    
    with open(f"history_{args.mode}.txt", "w") as f:
        for val in history:
            f.write(f"{val}\n")
            
if __name__ == "__main__":
    main()
