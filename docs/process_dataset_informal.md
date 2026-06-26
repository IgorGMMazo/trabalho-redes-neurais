# Explicacao do split do dataset pro grupo

O dataset original veio desbalanceado, com algumas pastas com mais de 2 mil fotos e outras com so 1000. Esse script resolve isso selecionando exatamente 1000 fotos de cada uma das 10 classes pra deixar o dataset balanceado, totalizando 10.000 imagens no total.

A divisao das pastas funciona assim:

1. O teste global (global_test)
Separa 15% de todas as fotos de cada classe logo no inicio. Da 150 fotos por classe, dando 1.500 no total. Essa pasta fica travada e nao eh usada durante o treino pra nao viciar o modelo, serve so pra testar a precisao final.

2. O pool de treino (os 85% restantes)
As 850 fotos de cada classe que sobraram sao divididas em duas metades identicas:
- A Fase 1 (train_fase_1): tem 425 fotos por classe.
- A Fase 2 (train_fase_2): tem 425 fotos por classe.

3. Os subsets da Fase 1 (escala logaritmica)
Cria pastas menores usando porcentagens pra simular cenarios onde temos poucos dados pra treinar. As porcentagens sao 1%, 2%, 5%, 10%, 20%, 50% e 100% da Fase 1.
Pela regra de inclusao cumulativa, as fotos da pasta de 1% estao dentro da de 2%, que estao dentro da de 5%, e assim por diante. Isso eh feito fazendo o fatiamento de uma lista embaralhada unica. Se a de 1% tem 4 fotos, a de 2% tem as mesmas 4 fotos mais 4 novas, totalizando 8.
A contagem de fotos por classe em cada subset:
- 1%: 4 fotos por classe (40 total)
- 2%: 8 fotos por classe (80 total)
- 5%: 21 fotos por classe (210 total)
- 10%: 42 fotos por classe (420 total)
- 20%: 85 fotos por classe (850 total)
- 50%: 212 fotos por classe (2.120 total)
- 100%: 425 fotos por classe (4.250 total)

4. O subset do futuro (fase_1_mais_fase_2)
Junta 100% da Fase 1 e 100% da Fase 2, totalizando 850 imagens por classe (8.500 total). Serve pra testar como a rede treina quando recebe mais dados depois de um tempo.

O arquivo dataset_metadata.json gerado no final guarda o caminho de onde cada foto foi parar pra provar que a divisao foi feita certa e com a seed 42.

Como rodar:

No computador local:
Rodar `python process_dataset.py` na pasta do projeto. Para testar se a logica ta certa sem copiar os arquivos e encher o disco, rodar `python process_dataset.py --dry_run`.

No Colab:
Subir o arquivo process_dataset.ipynb, colocar a pasta original la dentro e rodar as celulas na sequencia. No final ele ja compacta o resultado num zip.
