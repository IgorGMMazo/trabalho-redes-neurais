#!/usr/bin/env python3
"""
Gera e salva os pesos iniciais com herança progressiva.
Nenhum treinamento — só inicialização e transferência de pesos.

12.5% (Xavier puro) → herda → 25% → herda → 50% → herda → 100%

Saída em checkpoints/init/:
  init_12.5pct.pt
  init_25.0pct.pt
  init_50.0pct.pt
  init_100.0pct.pt
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from train import ScaledResNet18, MODEL_CONFIGS, apply_xavier_init, set_seed

CHECKPOINT_DIR = Path("checkpoints/init")
SEED_BASE      = 100
MODEL_SEQUENCE = [12.5, 25.0, 50.0, 100.0]


def transfer_weights(src_state: dict, target_model: nn.Module) -> None:
    target_state = target_model.state_dict()
    for key in target_state:
        if key not in src_state:
            continue
        src = src_state[key]
        dst = target_state[key]
        if src.shape == dst.shape:
            target_state[key].copy_(src)
        else:
            slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
            target_state[key][slices].copy_(src[slices])
    target_model.load_state_dict(target_state)


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GERAÇÃO DE PESOS INICIAIS COM HERANÇA PROGRESSIVA")
    print(f"  Saída: {CHECKPOINT_DIR.resolve()}")
    print("=" * 60)

    prev_state: Optional[dict] = None

    for i, model_size in enumerate(MODEL_SEQUENCE):
        cfg      = MODEL_CONFIGS[model_size]
        channels = cfg["channels"]

        set_seed(SEED_BASE + i)
        model = ScaledResNet18(channels=channels)
        apply_xavier_init(model)

        if prev_state is not None:
            transfer_weights(prev_state, model)
            print(f"  {model_size}%  — herdou pesos do {MODEL_SEQUENCE[i-1]}% + Xavier nos canais novos")
        else:
            print(f"  {model_size}%  — Xavier puro")

        ckpt_path = CHECKPOINT_DIR / f"init_{model_size}pct.pt"
        torch.save(
            {
                "model_size_pct": model_size,
                "channels":       channels,
                "alpha":          cfg["alpha"],
                "state_dict":     model.state_dict(),
            },
            ckpt_path,
        )
        print(f"           salvo em {ckpt_path}")

        prev_state = model.state_dict()

    print("\n[OK] 4 arquivos gerados.")


if __name__ == "__main__":
    main()
