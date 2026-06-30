# Documentação: `training_results.csv`

Este arquivo armazena os resultados detalhados das execuções do experimento de Leis de Escala (Scaling Laws) para o dataset de doenças foliares de tomate. Cada linha representa um registro de log capturado durante o treinamento de um modelo em um determinado subconjunto de dados.

## Definição das Colunas

| Coluna | Descrição |
| :--- | :--- |
| `model_size_pct` | Fração da capacidade do modelo ResNet-18 (ex: 12.5%, 25%, 50%, 100%) utilizada no experimento. Representa o tamanho do modelo (número de parâmetros). |
| `data_subset_pct` | Fração do conjunto de dados de treinamento disponível (ex: 1%, 2%, 5%, ..., 100%). Define o volume de dados utilizado em cada execução. |
| `step` | O passo de treinamento atual no momento da logagem (dentro do budget total definido). |
| `train_loss` | Valor da função de perda (Loss) no treinamento, calculada sobre o lote (batch) atual de dados. |
| `test_accuracy` | Acurácia calculada sobre o conjunto de teste fixo após o passo de treinamento especificado. |
| `test_error_rate` | Taxa de erro calculada como 1.0 - acurácia. Usada para análises de leis de escala (Power Laws). |
| `test_f1` | Métrica F1-Score (macro) no conjunto de teste, indicando o desempenho balanceado entre classes. |
| `test_precision` | Métrica de Precisão (macro) no conjunto de teste. |
| `elapsed_sec` | Tempo decorrido (em segundos) desde o início da execução daquele modelo e subset específico até o momento do registro. |