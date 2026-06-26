# Explicacao das Scaling Laws pro grupo

Esse script serve pra prever o erro do modelo sem precisar gastar tempo e processamento treinando ele no dataset completo da Fase 1+2. Pega os resultados dos treinos pequenos da Fase 1 (variando tamanho de dataset e capacidade da ResNet) e usa pra estimar qual vai ser o erro exato do modelo quando treinar na Fase 1 + Fase 2 (8.500 imagens).

O script compara duas previsoes:

1. A Lei de Rosenfeld
Eh uma equacao matematica que ajusta uma curva nos pontos de dados anteriores pra calcular o erro com base na capacidade da ResNet (M) e na quantidade de fotos (N). Ela calcula como o erro cai conforme os recursos de dados e largura do modelo aumentam.

2. Random Forest Regressor
Eh um algoritmo de machine learning que aprende a partir dos treinos anteriores, olhando o tamanho do dataset, a capacidade do modelo e a inclinacao da loss (loss_slope_final).

Validação e Gráficos:
- Rodando com a flag `--validate`, o script lê o erro real obtido no treino de Fase 1+2 e calcula a margem de erro da nossa previsao.
- O arquivo de imagem scaling_law_surface.png gerado no final mostra superficies em 3D de como o erro do modelo cai com o aumento do dataset e da capacidade. Serve pra colocar nos slides do trabalho.

Como rodar:

No computador local:
Rodar `python scaling_laws.py --results_csv training_results.csv --target_N 8500 --target_M 1.0`. Se quiser validar a previsao apos o treino real de Fase 1+2, rodar com `python scaling_laws.py --validate`.

No Colab:
Subir o notebook scaling_laws.ipynb, colocar o CSV gerado no treino e rodar as celulas. Os graficos 3D aparecem direto na tela.
