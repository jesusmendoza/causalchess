import torch
import torch.nn as nn
import sys
import os

sys.path.append('nnue-pytorch')
from model.model import NNUEModel
from model.config import ModelConfig
from model.quantize import QuantizationConfig

def run_tests():
    print("=== Iniciando Filtro de Sanidad (Sanity Checks) ===")
    
    # 1. Load dataset
    print("[1] Cargando dataset de prueba...")
    dataset = torch.load('nnuecc_dataset.pt')
    
    us = dataset['us'][:10]
    them = dataset['them'][:10]
    white_indices = dataset['white_indices'][:10]
    white_values = dataset['white_values'][:10]
    black_indices = dataset['black_indices'][:10]
    black_values = dataset['black_values'][:10]
    causal_embeddings = dataset['causal_embeddings'][:10]
    psqt_indices = dataset['psqt_indices'][:10]
    layer_stack_indices = dataset['layer_stack_indices'][:10]
    
    # 2. Init model
    print("[2] Inicializando NNUEModel...")
    config = ModelConfig()
    quantize_config = QuantizationConfig()
    model = NNUEModel(
        feature_name="HalfKAv2_hm^",
        config=config,
        quantize_config=quantize_config
    )
    model.eval()
    
    # 3. Test Zero-Impact (Forward pass without vs with zeroed embeddings)
    print("[3] Test de Impacto Cero (Zero-Impact Test)...")
    with torch.no_grad():
        out_baseline = model(
            us, them, white_indices, white_values, black_indices, black_values,
            psqt_indices, layer_stack_indices
        )
        out_injected = model(
            us, them, white_indices, white_values, black_indices, black_values,
            psqt_indices, layer_stack_indices, causal_embeddings=causal_embeddings
        )
        
    diff = torch.abs(out_baseline - out_injected).max().item()
    assert diff < 1e-6, f"¡ERROR! El modelo modificado altera la evaluación inicial. Diff: {diff}"
    print(f"  -> PASSED: Diff máxima = {diff}")
    
    # 4. Test Gradient Flow
    print("[4] Test de Flujo de Gradientes (Gradient Flow Test)...")
    model.train()
    
    # Force W_e to require gradients
    model.W_e.weight.requires_grad = True
    
    # We need inputs to require grad if we want to test flow through the model? No, just model weights.
    out_train = model(
        us, them, white_indices, white_values, black_indices, black_values,
        psqt_indices, layer_stack_indices, causal_embeddings=causal_embeddings
    )
    
    # Dummy loss
    target = torch.randn_like(out_train)
    loss = torch.nn.functional.mse_loss(out_train, target)
    loss.backward()
    
    grad_norm = model.W_e.weight.grad.norm().item()
    assert grad_norm > 0, "¡ERROR! Los gradientes no están fluyendo hacia W_e."
    print(f"  -> PASSED: Norma de gradientes en W_e = {grad_norm}")
    
    print("=== Todos los tests pasaron exitosamente. ¡Estamos cuerdos! ===")

if __name__ == "__main__":
    run_tests()
