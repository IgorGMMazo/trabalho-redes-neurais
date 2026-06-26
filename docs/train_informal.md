# Explicacao do codigo de treino pro grupo

Galera, fiz o codigo de treino do modelo em PyTorch. Vou explicar resumido aqui pra gente usar na apresentacao sem gaguejar.

Basicamente o script faz o treino da ResNet-18 do zero. Nao usamos pesos prontos da ImageNet pq o professor proibiu usar transfer learning nesse trabalho.

Aqui esta oque vcs precisam saber sobre o modelo e o treino:

1. Modificacao de largura do modelo (o parametro alpha)
Dá pra escolher qual a largura da nossa rede na hora de rodar. O script recalcula as convolucoes pra diminuir os parametros do modelo e a gente ver o impacto disso na precisao final:
- alpha 0.125: modelo minúsculo (12.5% do tamanho original)
- alpha 0.25: 25% de capacidade
- alpha 0.5: metade da capacidade
- alpha 1.0: modelo padrao da ResNet-18 (capacidade cheia)

2. Inicializacao Xavier e otimizador
Os pesos comecam inicializados com a tecnica Xavier Uniform pra evitar que o gradiente suma no comeco do treino. O otimizador eh o AdamW com learning rate em 2e-3 e weight decay em 1e-4 pra dar uma regularizada na loss.

3. Warmup cosseno e steps de treino
O professor pediu pra nao rodar por epocas, mas sim por steps (que sao as atualizacoes de pesos). O padrao sao 10 mil steps.
- Nos primeiros 5% de steps (500 passos se for 10 mil total) o learning rate sobe linearmente do zero ate o maximo.
- Nos outros 95% o learning rate cai seguindo um cosseno ate zerar de novo. Isso serve pro modelo convergir mais suavemente no final.
Ah e o batch size eh travado em 64. Nao colocamos data augmentation pras imagens pra nao enviesar os testes.

4. Como o teste eh feito
O modelo so eh avaliado no final de cada treino rodando em cima da pasta global_test. As metricas que a gente calcula sao a taxa de erro de teste (que eh 1 menos a acuracia top 1), o F1-score macro e o precision macro.
O script tambem calcula a inclinacao da curva de perda no fim do treino (loss_slope_final) analisando os ultimos 10% de steps pra gente provar pro professor se a perda continuava caindo ou ja tinha estabilizado.

5. Rodar 5 vezes seguidas
O script foi feito pra rodar o treino 5 vezes seguidas usando as seeds 42, 43, 44, 45 e 46 de forma automatica. No final de tudo ele cospe os resultados consolidados de cada seed num arquivo CSV chamado training_results.csv pra gente so copiar as metricas e colar no nosso slide.

Como colocar pra rodar:

No computador local:
python train.py --alpha 1.0 --steps 10000 --output_csv training_results.csv
Lembrando que o script eh inteligente e detecta se o PC tem GPU da NVIDIA (CUDA) ou placa da AMD (via DirectML), senao ele roda na CPU mesmo (so que ai demora bem mais).

No Colab:
So abrir o notebook train.ipynb que criei, ajustar o caminho da pasta de treino e teste nas variaveis e rodar a celula principal. Ele gera o CSV la dentro pra baixar.
