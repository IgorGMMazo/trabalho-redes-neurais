# Escalonamento Conjunto de Capacidade do Modelo e Volume de Dados em Redes Neurais Profundas

**Disciplina:** Redes Neurais  
**Dataset:** PlantVillage — 10 classes de doenças foliares de tomate (10.000 imagens, 1.000/classe)

---

## Pergunta de Pesquisa

Como o desempenho de um modelo de classificação de imagens varia conjuntamente em função da **capacidade do modelo** (número de parâmetros) e do **volume de dados de treinamento**, e existe uma relação de troca (trade-off) entre investir em mais dados versus em um modelo maior?

---

## Estrutura do Repositório

```
.
├── paper.ipynb                   ← NOTEBOOK PRINCIPAL (entrega final)
│                                   Contém toda a análise, gráficos e validação do meta-modelo
│
├── src/                          ← Código-fonte do pipeline
│   ├── split_dataset.py          ← Particionamento 70/15/15 com balanceamento estratificado
│   ├── init_weights.py           ← Geração e salvamento dos pesos iniciais fixos
│   ├── train.py                  ← Loop de treinamento com early stop e logging por steps
│   └── analyze_results.py        ← Ajuste da Power Law e geração dos gráficos de scaling
│
├── checkpoints/
│   └── init/                     ← Pesos iniciais fixos (garantem reprodutibilidade)
│       ├── init_12.5pct.pt       ← ResNet-18 com 12,5% da largura (~1,1M params)
│       ├── init_25.0pct.pt       ← ResNet-18 com 25% da largura (~4,3M params)
│       ├── init_50.0pct.pt       ← ResNet-18 com 50% da largura (~17M params)
│       └── init_100.0pct.pt      ← ResNet-18 com 100% da largura (~67M params)
│
├── resultados_earlystop/         ← Resultados do experimento principal (Grid Search 4×7)
│   ├── training_results.csv      ← Métricas de todos os 28 treinamentos (4 modelos × 7 subsets)
│   ├── scaling_law_plot.png      ← Gráfico geral da Power Law ajustada
│   ├── scaling_law_plot_12.5pct.png
│   ├── scaling_law_plot_25.0pct.png
│   ├── scaling_law_plot_50.0pct.png
│   ├── scaling_law_plot_100.0pct.png
│   └── scaling_law_plot_heatmap.png  ← Heatmap de acurácia (modelo × volume de dados)
│
├── resultados_intermediarios/    ← Resultados dos pontos ocultos (validação do meta-modelo)
│   ├── training_results.csv      ← Métricas dos 20 treinamentos nos subsets 3/7/15/30/70%
│   ├── scaling_law_plot.png
│   ├── scaling_law_plot_12.5pct.png
│   ├── scaling_law_plot_25.0pct.png
│   ├── scaling_law_plot_50.0pct.png
│   ├── scaling_law_plot_100.0pct.png
│   └── scaling_law_plot_heatmap.png
│
└── requirements.txt              ← Dependências Python
```

---

## Como Executar

```bash
pip install -r requirements.txt

# 1. Particionar o dataset (requer dataset original em dataset/preprocessed/)
python src/split_dataset.py

# 2. Gerar pesos iniciais fixos
python src/init_weights.py

# 3. Treinar — Grid Search (4 modelos × subsets 1,2,5,10,20,50,100%)
python src/train.py \
    --data-dir "dataset/preprocessed" \
    --output-csv "resultados_earlystop/training_results.csv" \
    --total-steps 10000 --batch-size 64 --lr 0.002 \
    --model-sizes "12.5,25,50,100" \
    --subsets "1,2,5,10,20,50,100"

# 4. Treinar pontos intermediários (validação meta-modelo)
python src/train.py \
    --data-dir "dataset/preprocessed" \
    --output-csv "resultados_intermediarios/training_results.csv" \
    --total-steps 10000 --batch-size 64 --lr 0.002 \
    --model-sizes "12.5,25,50,100" \
    --subsets "3,7,15,30,70"

# 5. Análise e gráficos de Scaling Law
python src/analyze_results.py

# 6. Abrir o notebook para ver toda a análise
jupyter notebook paper.ipynb
```

---

## Metodologia em Resumo

| Aspecto | Detalhe |
|---|---|
| **Arquitetura** | ResNet-18 *from scratch* com multiplicador de largura α ∈ {12.5%, 25%, 50%, 100%} |
| **Volumes de treino** | 1%, 2%, 5%, 10%, 20%, 50%, 100% do pool de treinamento (7.000 imagens) |
| **Critério de parada** | Early stop: melhoria < 1% a cada 500 steps (budget máximo: 10.000 steps) |
| **Pesos iniciais** | Fixos por tamanho de modelo (seeds 42–45), garantindo comparação justa |
| **Meta-modelo** | Power Law $E(x) = a \cdot x^{-b} + c$ ajustada nos pontos ≤ 50%, extrapolada para 100% |
| **Validação** | Pontos ocultos em 3%, 7%, 15%, 30%, 70% (nunca vistos pelo meta-modelo) |
| **Split** | 70% treino / 15% intermediário / 15% teste fixo |

---

## Onde Encontrar Cada Coisa no Notebook (`paper.ipynb`)

| Seção | O que contém |
|---|---|
| Pergunta / Hipótese | Motivação e formalização do problema |
| Metodologia | Dataset, arquitetura, particionamento, treinamento |
| Células de execução | Chamadas aos scripts `src/` |
| Análise exploratória | Gráficos de acurácia e loss por volume de dados |
| Meta-modelo (Power Law) | Ajuste da curva, previsão para 100%, gráfico por modelo |
| **Validação** | **Comparação Real vs Previsto: pontos intermediários + 100%** |
| | Subplots por modelo, scatter real×previsto, barras de erro, heatmap |
| Conclusão | Discussão dos resultados à luz da literatura (Rosenfeld 2020) |

---

## Referências

- Rosenfeld et al. (2020) — *A Constructive Prediction of the Generalization Error Across Scales*, ICLR
- He et al. (2016) — *Deep Residual Learning for Image Recognition*, CVPR
- Zagoruyko & Komodakis (2016) — *Wide Residual Networks*, BMVC
- Li et al. (2018) — *Visualizing the Loss Landscape of Neural Nets*, NeurIPS
