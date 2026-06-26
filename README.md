# Tomato Leaf Classification & Scaling Laws Pipeline

Este repositório contém o pipeline completo para preparação de dados, treinamento de modelos e modelagem de Leis de Escala (Scaling Laws) para o dataset de folhas de tomate.

---

## 📁 Estrutura do Projeto

Abaixo está o mapeamento da árvore de diretórios do repositório:

```
trabalho-redes-neurais/
├── dataset/                  <-- Dados de entrada (originais) e processados
│   ├── orginal/
│   └── processed/
├── docs/                     <-- Relatórios e documentações de suporte
│   ├── process_dataset.md    (Documentação técnica do dataset)
│   ├── process_dataset_informal.md  (Explicação informal do dataset)
│   ├── train.md              (Documentação técnica do treino)
│   ├── train_informal.md     (Explicação informal do treino)
│   ├── scaling_laws.md       (Documentação técnica das Scaling Laws)
│   ├── scaling_laws_informal.md  (Explicação informal das Scaling Laws)
│   ├── main.md               (Documentação técnica do orquestrador)
│   └── main_informal.md      (Explicação informal do orquestrador)
├── notebooks/                <-- Códigos Jupyter Notebook para o Google Colab
│   ├── pipeline_completo.ipynb (Unificado com textos explicativos)
│   ├── process_dataset.ipynb
│   ├── train.ipynb
│   ├── scaling_laws.ipynb
│   └── colab_consolidated_cell.py
├── requirements.txt          <-- Dependências de pacotes Python
├── process_dataset.py        <-- Script CLI de particionamento do dataset
├── train.py                  <-- Script CLI de treinamento em PyTorch
├── scaling_laws.py           <-- Script CLI de regressão e predição de erros
└── main.py                   <-- Orquestrador automático de experimentos (Grid Search)
```

---

## 📊 Estrutura e Divisão do Dataset

O pipeline executa um balanceamento estrito selecionando exatamente **1.000 imagens por classe** (totalizando 10.000 imagens nas 10 classes) a partir do diretório `dataset/orginal/`. Esses dados são divididos e salvos em `dataset/processed/` seguindo a seguinte estrutura:

1. **Global Test Set (`dataset/processed/global_test/`)**:
   * **Proporção:** 15% das imagens de cada classe (1.500 imagens no total, 150 por classe).
   * **Finalidade:** Conjunto de teste final (intocável durante o treinamento e usado para calcular o erro real).

2. **Train Pool (85% restantes - 8.500 imagens no total)**:
   * **Fase 1 (`dataset/processed/train_fase_1/`):** Metade do conjunto de treino (4.250 imagens, 425 por classe). Usado para o Grid Search de Scaling Laws.
   * **Fase 2 (`dataset/processed/train_fase_2/`):** Segunda metade do conjunto de treino (4.250 imagens, 425 por classe).
   * **Fase 1 + 2 (`dataset/processed/fase_1_mais_fase_2/`):** União de ambas as fases (8.500 imagens). Usado como validação empírica da extrapolação de Scaling Laws.

3. **Subsets Cumulativos da Fase 1 (`dataset/processed/subsets_fase_1/`)**:
   Divisões logarítmicas da Fase 1 usadas no Grid Search para calibrar as curvas de Scaling Laws. Obedece à *Regra de Inclusão Cumulativa* (o subset de 1% está contido no de 2%, que está contido no de 5%, etc.):
   * **Subset 1%:** 40 imagens (4 por classe)
   * **Subset 2%:** 80 imagens (8 por classe)
   * **Subset 5%:** 210 imagens (21 por classe)
   * **Subset 10%:** 420 imagens (42 por classe)
   * **Subset 20%:** 850 imagens (85 por classe)
   * **Subset 50%:** 2.120 imagens (212 por classe)
   * **Subset 100%:** 4.250 imagens (425 por classe)

---

## 🚀 Como Executar

### 1. Orquestração Completa (Grid Search de 140 Execuções + Extrapolação + Validação)
Roda de forma automática todas as fases de preparação, treino multisseed dos subsets, regressão, predição e teste de validação:
```bash
python main.py --steps 1000 --target_steps 10000
```

### 2. Execuções Individuais
- **Preparar Dataset**: `python process_dataset.py`
- **Treinar Modelo**: `python train.py --alpha 1.0 --steps 10000`
- **Predizer Scaling Laws**: `python scaling_laws.py --target_N 8500 --target_M 1.0`
