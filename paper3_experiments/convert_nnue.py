import sys
sys.path.append('nnue-pytorch')
import torch
from model.config import NNUELightningConfig
from model.lightning_module import NNUE
from model.utils.serialize import NNUEReader
import argparse

from model.quantize import QuantizationConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    config = NNUELightningConfig()
    model = NNUE(config=config, max_epoch=1, num_batches_per_epoch=1)

    with open(args.input, 'rb') as f:
        reader = NNUEReader(f, config.features, config.model_config, QuantizationConfig())
        model.model = reader.model
    
    # We must ensure that W_e exists because our NNUE adds it!
    # The reader's model might not have W_e if it instantiates its own architecture,
    # or it uses the class that has W_e but leaves it uninitialized.
    # Let's just save the LightningModule state dict
    torch.save(model.state_dict(), args.output)
    print(f"Successfully converted {args.input} to {args.output}")

if __name__ == '__main__':
    main()
