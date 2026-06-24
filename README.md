# Tomato Leaf Classification & Scaling Law Analysis Pipeline

Este repositório contém a suíte completa de engenharia de dados, treinamento de modelos e análise de Scaling Laws desenvolvida para o trabalho de Redes Neurais.

---

## 📂 Estrutura do Projeto Reorganizada

Para melhor legibilidade e organização acadêmica, o repositório está dividido nas seguintes pastas lógicas:

```text
trabalho-redes-neurais/
│
├── dataset/                        # Base de dados (Ignorada no Git)
│   ├── orginal/                    # Imagens originais por classe
│   └── preprocessed/               # Subsets logarítmicos gerados
│
├── docs/                           # Explicações e Documentações (Formal/Informal)
│   ├── split_dataset.md            # Documentação técnica do split
│   ├── sliptt_dataset_explicação_informal.md  # Explicação informal (grupo) do split
│   ├── train_dataset.md            # Documentação técnica do treinamento
│   ├── train_dataset_explicação_informal.md  # Explicação informal do treinamento
│   ├── analyze_results.md          # Documentação técnica da regressão logarítmica
│   └── analyze_results_explicação_informal.md  # Explicação informal da regressão logarítmica
│
├── notebooks/                      # Jupyter Notebooks para o Google Colab
│   ├── preprocess_colab.ipynb      # Pipeline de split e amostragem cumulativa
│   ├── train_colab.ipynb           # Loop de treinamento com seeds
│   └── analyze_results_colab.ipynb # Célula de análise estatística e plotagem
│
├── results/                        # Logs de Treinamento e Gráficos gerados
│   ├── training_results.csv        # Log incremental de épocas
│   └── scaling_law_plot.png        # Gráfico gerado de curva de aprendizado
│
├── src/                            # Scripts Python locais (CLI via argparse)
│   ├── split_dataset.py            # Pré-processamento e balanceamento
│   ├── train.py                    # Treinamento com suporte multi-hardware
│   └── analyze_results.py          # Ajuste de curva e regressão
│
├── .gitignore
└── requirements.txt                # Dependências do projeto
```

---

## 🚀 Como Executar Localmente

### 1. Instale as dependências:
```bash
pip install -r requirements.txt
```

### 2. Prepare o Dataset (Balanceamento + Subsets Cumulativos):
```bash
python src/split_dataset.py
```
*Gera os subsets cumulativos `[1%, 2%, 5%, 10%, 20%, 50%, 100%]` na pasta `dataset/preprocessed/`.*

### 3. Execute os Treinamentos (ResNet-18 do Zero com 5 seeds por subset):
```bash
python src/train.py
```
*Salva os logs de loss de treino, acurácia e precisão de teste por época no arquivo `results/training_results.csv`.*

### 4. Ajuste as Scaling Laws e Plote a Curva Prevista:
```bash
python src/analyze_results.py
```
*Usa pandas/numpy/matplotlib para gerar o relatório de erros de extrapolação e o gráfico `results/scaling_law_plot.png`.*

---

## 📝 Documentações de Apoio (docs/)
Consulte a pasta [docs](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/docs) para explicações detalhadas em formato formal e linguagem informal de suporte ao grupo.
