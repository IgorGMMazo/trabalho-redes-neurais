# Como funciona a análise de Scaling Laws (pra galera do grupo)

Fiz esse script de análise (analyze_results.py) e o notebook do Colab (analyze_results_colab.ipynb) pra gente fazer o ajuste de curvas que a professora pediu pra prever os resultados dos treinos futuros. Segue o resumo pra gente alinhar a apresentação.

## O que é esse ajuste de curva (Scaling Law)?
A ideia é bem simples: treinar modelos grandes ou com muitas fotos demora horas e custa caro. Então a gente quer ver se da pra prever o resultado final (Loss de treino e Acurácia de teste) usando só os resultados dos subsets pequenos (tipo 1%, 2% e 5%), sem ter que treinar os maiores pra saber o resultado.

Esse script faz exatamente isso usando uma regressão logarítmica (ajuste matemático do tipo `y = a * ln(x) + b`).

## Como funciona o script
1. **Lê o CSV de Métricas**: Ele puxa o arquivo `training_results.csv` que a gente gerou no treino anterior.
2. **Pega a Última Época**: Ele filtra as métricas da última época de cada rodada e tira a média entre as 5 seeds pra ter um dado bem estável.
3. **Faz Previsões Sequenciais**: 
   - Ele pega os dados de 1% e 2% pra prever quanto seria no de 5% e compara com o que a gente obteve na vida real.
   - Pega 1%, 2% e 5% pra prever o de 10% e calcula o erro.
   - E vai fazendo isso pra todos os passos, calculando o Erro Absoluto e o Erro Relativo (%) em cada etapa.
4. **Plota o Gráfico**: Ele gera uma figura chamada `scaling_law_plot.png` mostrando os pontinhos vermelhos/azuis reais e a linha pontilhada da previsão matemática. Isso vai ficar perfeito pra gente colar nos slides do trabalho.

## Como rodar local no PC
Pra rodar o ajuste usando as 3 primeiras pastas (1%, 2%, 5%) pra desenhar a previsão pro resto:
```bash
python analyze_results.py
```
*(Se vc quiser ajustar usando mais ou menos pontos, pode passar o argumento `--fit-points 4` por exemplo).*

Se não tiver o matplotlib instalado no seu python local, ele só vai cuspir as tabelas de erros no terminal e pular a parte do gráfico pra não travar a execução.

No Colab, é só abrir o notebook `analyze_results_colab.ipynb` e rodar a célula de análise pra plotar o gráfico inline direto no navegador.
