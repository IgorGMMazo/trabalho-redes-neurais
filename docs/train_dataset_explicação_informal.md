# Como funciona o treino das redes (pra galera do grupo)

Fiz esse script de treino (train.py) e o notebook do Colab (train_colab.ipynb) pra rodar o treinamento da ResNet-18 nas pastas que a gente gerou. Segue a explicação direta de como funciona pra gente nao se embananar na entrega.

## A detecção de placa de vídeo (Hardware)
Pra ninguem ter erro de rodar em computadores diferentes, o script detecta a GPU de forma automatica:
- Se vc tiver placa NVIDIA (ou rodar no Colab com GPU ativa), ele usa CUDA.
- Se vc tiver placa AMD (Radeon) no Windows, ele tenta usar DirectML (só precisa ter o pacote `torch-directml` instalado).
- Se nao tiver placa nenhuma, ele vai rodar na CPU (vai demorar mais, mas roda).

Ele mostra qual placa detectou bem no começo do terminal pra gente saber se ta usando aceleração ou nao.

## Sem Data Augmentation
A gente nao ta usando nenhuma alteração louca nas fotos (tipo girar ou cortar). Só fazemos o redimensionamento pra 224x224 (que é o tamanho padrão da ResNet-18) e a normalização pros valores ficarem na mesma escala. A unica coisa que muda de um treino pro outro sao os pesos iniciais da rede.

## Por que treinar 5 vezes por subset?
O professor pediu pra testar a estabilidade do treino. Entao, pra cada subset (1%, 2%, 5%, 10%, 20%, 50% e 100%), o script roda o treinamento completo **5 vezes seguidas**, cada uma usando um seed (semente) diferente: `42, 43, 44, 45, 46`. 

Isso reinicializa os pesos iniciais da rede de forma diferente pra gente avaliar se o modelo aprende igual ou se da muita variação nas métricas.

## Salvamento dos Resultados (CSV)
Toda época de cada rodada é avaliada e salva na hora no arquivo `training_results.csv`. O script salva:
- A loss de treino.
- A acurácia no grupo de teste.
- A precisão (macro-average) no grupo de teste.

## Como rodar o treino local
Pra rodar a parada padrão (10 epocas, batch size 32):
```bash
python train.py
```
Se quiser rodar rapido pra testar se ta tudo funcionando na sua maquina antes do treino real:
```bash
python train.py --subsets 1 --epochs 1 --runs-per-subset 1
```
Isso vai rodar só uma época no subset de 1% e ja cria o CSV pra vc ver se ta gravando tudo certinho.
