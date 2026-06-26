# Explicacao do orquestrador main.py pro grupo

Esse script serve pra rodar todos os passos do nosso trabalho de uma vez so de forma automatica, sem a gente precisar ficar chamando script por script no terminal toda hora.

Aqui esta oque ele faz passo a passo na execucao:

1. Particionamento do Dataset
Chama o script process_dataset.py pra limpar as pastas antigas e criar de novo os splits balanceados com 1000 fotos de tomate por classe.

2. Execucao do Grid Search (140 treinos)
Roda o loop principal de testes combinando:
- As 4 capacidades de tamanho da ResNet (12.5%, 25%, 50% e 100% de alpha).
- Os 7 subconjuntos de tamanho de dados da Fase 1 (1%, 2%, 5%, 10%, 20%, 50% e 100%).
Como sao 28 combinacoes e cada uma roda com 5 seeds diferentes, o script executa 140 rodadas de treino completas no train.py de forma sequencial. Os logs de cada treino sao printados com um recuo de linha pra gente conseguir ler o progresso. Todos os dados sao acumulados no CSV training_results.csv.

3. Extrapolacao das Scaling Laws
Chama o script scaling_laws.py pra rodar a equacao matematica em cima do CSV gerado no passo anterior. Ele calcula os coeficientes e cospe no console a predicao cega de qual seria o erro do modelo se a gente usasse 100% dos dados da Fase 1+2 (8.500 imagens) com a ResNet de capacidade cheia (alpha 1.0).

4. Treino de Validacao Real
Roda um treinamento real usando a pasta fase_1_mais_fase_2 (8.500 imagens) com 100% de capacidade do modelo. O resultado do erro real eh salvo num CSV separado chamado validation_results.csv.

5. Cruzamento e Margem de Erro
Chama a validacao do scaling_laws.py pra cruzar os dois dados. Ele compara o erro real do passo 4 com as previsoes matematicas que calculamos no passo 3, mostrando a margem de erro exata da nossa predicao no final da tela.

Como rodar:

No computador local:
Roda `python main.py --steps 1000 --target_steps 5000` pra executar o fluxo completo. Dá pra usar a flag `--dry_run` pra ver a lista de comandos no terminal sem rodar nada.

No Colab:
Tem o arquivo colab_consolidated_cell.py. Dá pra copiar todo o conteudo dele e colar em uma unica celula do notebook no Colab pra rodar todo o pipeline de ponta a ponta com GPU.
