# Pipeline de Processamento de Imagens para Visão Computacional

Este documento descreve as especificações técnicas e de engenharia de dados do pipeline de pré-processamento implementado para o dataset de folhas de tomate composto por 10 classes.

---

## 🛠️ Especificações de Engenharia de Dados

### 1. Balanceamento Inicial Rígido
O dataset de entrada pode apresentar desbalanceamento natural devido à amostragem. Para mitigar o viés do modelo induzido por distribuições de classes desbalanceadas, o script realiza um truncamento rígido inicial:
- Seleciona **exatamente 1.000 imagens por classe** a partir da ordenação alfabética inicial e shuffle reprodutível.
- Tamanho total do dataset controlado: **10.000 imagens**.

### 2. Global Test Set (Estratificação Estrita)
- **Proporção**: 15% das imagens originais selecionadas de cada classe.
- **Volume**: 150 imagens por classe (1.500 imagens no total).
- **Diretório**: `dataset/processed/global_test/`
- **Garantia**: Este conjunto é isolado fisicamente antes de qualquer outra partição e permanece inalterável para avaliação final de performance dos modelos.

### 3. Particionamento do Global Train Pool (85% restantes)
As 850 imagens restantes por classe são divididas em duas frações exatamente iguais de 50% cada:
1. **Fase 1 (`train_fase_1`)**: 425 imagens por classe (4.250 no total).
2. **Fase 2 (`train_fase_2`)**: 425 imagens por classe (4.250 no total).

### 4. Subsets Cumulativos da Fase 1 (Regra de Inclusão Cumulativa)
Para simular restrições de escala de dados em experimentos de aprendizagem ativa (Active Learning) ou curvas de aprendizado, foram gerados subsets cumulativos em escala logarítmica sobre a Fase 1: `[1%, 2%, 5%, 10%, 20%, 50%, 100%]`.

A **Regra de Inclusão Cumulativa** garante que:
$$S_{1\%} \subset S_{2\%} \subset S_{5\%} \subset S_{10\%} \subset S_{20\%} \subset S_{50\%} \subset S_{100\%}$$

Isso é obtido fatiando deterministicamente os primeiros $N$ elementos de uma lista ordenada e embaralhada única da Fase 1:
- `pct_1` (1%): 4 imagens/classe (total 40)
- `pct_2` (2%): 8 imagens/classe (total 80)
- `pct_5` (5%): 21 imagens/classe (total 210)
- `pct_10` (10%): 42 imagens/classe (total 420)
- `pct_20` (20%): 85 imagens/classe (total 850)
- `pct_50` (50%): 212 imagens/classe (total 2.120)
- `pct_100` (100%): 425 imagens/classe (total 4.250)

### 5. Subset do Futuro (`fase_1_mais_fase_2`)
Conjunto que consolida 100% de `train_fase_1` e 100% de `train_fase_2` em uma única pasta física para fins de simulação de novos ciclos de treinamento.
- **Volume**: 850 imagens por classe (8.500 imagens no total).

### 6. Rastreabilidade de Linhagem (`dataset_metadata.json`)
O arquivo de metadados gerado armazena o caminho original, o split de destino correspondente, a classe, e todas as réplicas criadas para cada arquivo.

---

## 💻 Instruções de Execução

### Script Python Local (`process_dataset.py`)
Certifique-se de que os pacotes básicos estejam instalados. Execute o script passando os caminhos ou aceitando os defaults do projeto:
```bash
python process_dataset.py --input_dir "dataset/orginal/dataset-plantas-com-augmentation" --output_dir "dataset/processed" --seed 42 --max_images_per_class 1000
```

### Notebook Google Colab (`process_dataset.ipynb`)
1. Carregue o arquivo no ambiente do Colab.
2. Certifique-se de descompactar o dataset original no diretório configurado em `INPUT_DIR`.
3. Execute todas as células. O pipeline gerará o arquivo `tomato_processed_dataset.zip` no diretório raiz do Colab para download facilitado.
