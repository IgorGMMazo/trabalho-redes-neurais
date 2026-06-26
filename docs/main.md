# Orquestrador Automático de Experimentos (main.py)

Este documento descreve a arquitetura técnica e o fluxo de controle do script `main.py`, que atua como o orquestrador central do pipeline de experimentos para validação de Scaling Laws.

---

## ⚙️ Fluxo de Controle e Execução

O script gerencia a execução sequencial e paralela-bloqueante de processos filhos (subprocessos) de modo a garantir a consistência das entradas e saídas do pipeline.

```
                  ┌──────────────────────┐
                  │ 1. process_dataset   │ (Executa split e balanceamento)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ 2. Grid Search (140) │ (4 alphas x 7 subsets x 5 seeds)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ 3. Fit scaling_laws  │ (Ajusta Rosenfeld e Random Forest)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ 4. Ground Truth      │ (Treina Fase 1+2 c/ alpha 1.0)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ 5. Validation        │ (Calcula erro de predição cega)
                  └──────────────────────┘
```

---

## 🛠️ Especificações Técnicas de Implementação

### 1. Gerenciamento de Processos via Subprocessos
O orquestrador instancia cada etapa como um processo separado usando a biblioteca `subprocess.Popen` do Python. Isso garante:
- **Isolamento de Memória**: Previne fragmentação de memória ou vazamento de cache do PyTorch (CUDA memory leaks) entre os 140 treinamentos individuais.
- **Log Assíncrono Dinâmico**: O stdout de cada subprocesso é capturado em tempo de execução e formatado sob o prefixo `│` no terminal cyber-industrial do orquestrador.

### 2. Parâmetros Editáveis CLI
- `--steps`: Steps de treinamento para cada uma das 140 execuções do Grid Search (default: `1000`).
- `--target_steps`: Steps de treinamento para a avaliação final com 100% de capacidade e dados (default: `10000`).
- `--img_size`: Dimensão de redimensionamento da imagem nas etapas de carregamento (default: `128`).
- `--max_images_per_class`: Número limite de imagens selecionadas por classe para fins de balanceamento (default: `1000`).
- `--dry_run`: Flag de validação estrutural que exibe todos os comandos que seriam invocados sem executá-los de fato.
