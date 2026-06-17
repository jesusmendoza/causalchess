import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import argparse
import os
import glob
import gc

sys.path.append('nnue-pytorch')
from model.config import NNUELightningConfig
from model.lightning_module import NNUE

def load_chunk_dataloaders(chunk_path, batch_size=512):
    print(f"  -> Cargando {os.path.basename(chunk_path)}...", end=" ")
    data = torch.load(chunk_path)
    dataset = TensorDataset(
        data['us'], data['them'], data['white_indices'], data['white_values'],
        data['black_indices'], data['black_values'], data['outcome'],
        data['score'], data['psqt_indices'], data['layer_stack_indices'],
        data['causal_embeddings']
    )
    
    total_size = len(dataset)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"({train_size} train / {val_size} val)")
    
    return train_loader, val_loader

def eval_loop(model, val_loader, mode):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = [t.to(model.device) for t in batch]
            if mode == "baseline":
                batch[-1] = torch.zeros_like(batch[-1])
            loss = model.step_(batch, 0, "val_loss")
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_epoch(model, train_loader, optimizer, mode):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = [t.to(model.device) for t in batch]
        if mode == "baseline":
            batch[-1] = torch.zeros_like(batch[-1])
        optimizer.zero_grad()
        loss = model.step_(batch, 0, "train_loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def process_epoch_chunks(model, chunk_files, optimizer, mode, is_train=True):
    total_loss = 0.0
    total_batches = 0
    
    for chunk_file in chunk_files:
        train_loader, val_loader = load_chunk_dataloaders(chunk_file)
        
        if is_train:
            loss = train_epoch(model, train_loader, optimizer, mode)
            batches = len(train_loader)
        else:
            loss = eval_loop(model, val_loader, mode)
            batches = len(val_loader)
            
        total_loss += loss * batches
        total_batches += batches
        
        # Limpieza de RAM tras procesar el chunk
        del train_loader, val_loader
        gc.collect()
        
    return total_loss / total_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "nnuecc"], required=True)
    parser.add_argument("--input_dir", default="dataset_15M")
    parser.add_argument("--epochs_warmup", type=int, default=1)
    parser.add_argument("--epochs_finetune", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = f"best_model_{args.mode}.pt"

    print(f"=== Entrenamiento Chunked para Competencia: {args.mode.upper()} ===")
    print(f"Device: {device}")

    chunk_files = sorted(glob.glob(os.path.join(args.input_dir, "chunk_*.pt")))
    if not chunk_files:
        print(f"Error: No se encontraron archivos chunk_*.pt en {args.input_dir}")
        return
        
    print(f"Se encontraron {len(chunk_files)} chunks en {args.input_dir}")

    total_epochs = args.epochs_warmup + args.epochs_finetune
    
    config = NNUELightningConfig()
    config.loss_params.start_lambda = 1.0
    config.loss_params.end_lambda = 1.0

    model = NNUE(
        config=config,
        max_epoch=total_epochs,
        num_batches_per_epoch=1000, # Approximate value, not strictly needed for basic Adam
    ).to(device)

    if args.resume:
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"[!] Resumed from {checkpoint_path}")
        except Exception as e:
            print(f"Could not resume from {checkpoint_path}: {e}")

    val_history = []
    best_val_loss = float('inf')

    # === STAGE 1: WARM-UP ===
    if args.mode == "nnuecc":
        print(f"\n[STAGE 1] Warm-up W_e solamente | LR=1e-4 | {args.epochs_warmup} epochs")
        for name, param in model.named_parameters():
            param.requires_grad = ("W_e" in name)
    else:
        print(f"\n[STAGE 1] Warm-up Base (sin W_e) | LR=1e-4 | {args.epochs_warmup} epochs")
        for name, param in model.named_parameters():
            param.requires_grad = ("W_e" not in name)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    for epoch in range(args.epochs_warmup):
        print(f"--- Epoch {epoch+1}/{args.epochs_warmup} (Warm-Up) ---")
        train_loss = process_epoch_chunks(model, chunk_files, optimizer, args.mode, is_train=True)
        val_loss = process_epoch_chunks(model, chunk_files, optimizer, args.mode, is_train=False)
        
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            marker = " <- BEST"
        else:
            marker = ""

        print(f"  Resultados E{epoch+1:2d}: train={train_loss:.5f} | val={val_loss:.5f}{marker}\n", flush=True)

    # === STAGE 2: DEEP FINE-TUNING ===
    print(f"\n[STAGE 2] Deep Fine-Tune (all layers) | LR=1e-5 | {args.epochs_finetune} epochs")
    for name, param in model.named_parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )

    for epoch in range(args.epochs_finetune):
        print(f"--- Epoch {epoch+1}/{args.epochs_finetune} (Fine-Tune) ---")
        train_loss = process_epoch_chunks(model, chunk_files, optimizer, args.mode, is_train=True)
        val_loss = process_epoch_chunks(model, chunk_files, optimizer, args.mode, is_train=False)
        
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            marker = " <- BEST"
        else:
            marker = ""

        print(f"  Resultados E{epoch+1:2d}: train={train_loss:.5f} | val={val_loss:.5f}{marker}\n", flush=True)

    print(f"\n=== Entrenamiento Chunked Finalizado ===")
    print(f"Mejor Val Loss ({args.mode}): {best_val_loss:.5f}")
    print(f"Checkpoint: {checkpoint_path}")

    with open(f"history_val_{args.mode}.txt", "w") as f:
        for val in val_history:
            f.write(f"{val}\n")

if __name__ == "__main__":
    main()
