# Como funciona a divisão do dataset (pra galera do grupo)

Fiz esse script com a preparação das imagens pras redes neurais. Vou resumir aqui como funciona pra ninguem se perder na hora de rodar ou na hora de explicar pro professor.

## O problema do dataset original
O dataset original ta desbalanceado. Tem classe la com mais de 5000 fotos e outras com 1000. Se a gente treinar a rede assim vai dar ruim porque ela vai enviesar nas classes com mais fotos.

Por isso, a primeira coisa que o script faz é limitar todas as 10 classes pra no maximo 1000 imagens cada. Ele embaralha e pega as 1000 primeiras. No final o nosso dataset base (o de 100%) fica com exatamente 10.000 imagens no total.

## Divisão de Treino e Teste (Split)
Pra cada classe, o script separa o treino do teste na proporção de 85% e 15%. Ou seja, das 1000 fotos de cada classe:
- 850 vao pra treino
- 150 vao pra teste

Esse split é feito logo de cara e fica fixo. Isso é super importante pra regra cumulativa.

## Regra de inclusão cumulativa
A gente precisa gerar os subsets com as porcentagens [1%, 2%, 5%, 10%, 20%, 50%, 100%]. 
Pra respeitar a regra de que o subset maior tem que conter o menor, a lógica funciona por fatiamento (slice):
- O de 1% pega as primeiras 10 imagens (8 de treino, 2 de teste) de cada classe.
- O de 2% pega as primeiras 20 imagens. Essas 20 contem exatamente as 10 fotos que ja estavam no de 1%, mais 10 fotos novas do pool.
- O de 5% pega as primeiras 50 imagens (as 20 anteriores + 30 novas).
- E assim vai ate o de 100%.

Como o pool de treino e teste de cada classe é fixado no inicio, a gente garante que uma imagem que é de treino em 1% continua sendo de treino em 2%, 5%, etc., e nunca vai parar na pasta de teste de outra porcentagem. Isso evita vazamento de dados (data leakage) e garante a regra cumulativa.

## Pastas geradas
O script cria tudo dentro de 'dataset/preprocessed'. La dentro ele cria uma pasta pra cada subset (subset_1%, subset_2%, etc).
Dentro de cada subset tem as pastas 'train' e 'test', e dentro delas as pastas das doenças com as fotos copiadas.

Tambem gera o arquivo 'dataset_metadata.json' que rastreia tudo. Se a gente precisar puxar exatamente quais fotos foram pra cada subset e split, ta tudo salvo nesse json.

## Como rodar
Pra rodar local no PC:
python split_dataset.py

E se for rodar no Colab pra treinar, criei o notebook 'preprocess_colab.ipynb'. É so abrir la, rodar a celula pra montar o Google Drive e executar o pipeline. Ja deixei os caminhos relativos prontos pra nao ter erro de caminho no Drive de vcs.
