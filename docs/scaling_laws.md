# Modelagem de Scaling Laws (Lei de Rosenfeld e Random Forest)

Este documento descreve a metodologia e a base teórica utilizadas para modelar as Leis de Escala (Scaling Laws) e realizar extrapolações de performance no dataset de folhas de tomate.

---

## 📐 Fundamentação Teórica

### 1. Ajuste Teórico Paramétrico (Equação de Rosenfeld)
As Leis de Escala em Deep Learning descrevem como a taxa de erro de teste decai em função do tamanho do dataset ($N$) e da capacidade paramétrica do modelo ($M$). 
Utilizou-se a formulação proposta por Rosenfeld (2020), que modela a interação conjunta desses fatores:
$$\varepsilon(M, N) = a \cdot M^{-\alpha} + b \cdot N^{-\beta} + c_\infty$$

Onde:
- **$\varepsilon$**: Taxa de erro de teste esperada.
- **$M$**: Fração de capacidade do modelo (relação de parâmetros / $\alpha$).
- **$N$**: Volume total de imagens do dataset de treinamento.
- **$c_\infty$**: Limite de erro assintótico irredutível do modelo.
- **$\alpha, \beta$**: Expoentes de escala que determinam o decaimento do erro conforme o aumento de recursos.

O ajuste é feito pelo método de mínimos quadrados não lineares com restrições (`scipy.optimize.curve_fit`).

### 2. Regressão Empírica Não Paramétrica (Random Forest)
Em paralelo ao ajuste paramétrico, o script emprega um regressor baseado em florestas de decisão (`RandomForestRegressor`). Este modelo mapeia o vetor de entrada $[N, M, \text{loss\_slope\_final}]$ diretamente para a taxa de erro de teste ($\varepsilon$), capturando relações e saturações não-lineares arbitrárias no espaço amostral do treino.

---

## 🚀 Como Executar

### Predição Cega de Extrapolação (Fase 1 + Fase 2)
Para predizer o erro de teste que o modelo obteria ao ser treinado no dataset consolidado de 8.500 imagens ($N = 8500$) usando 100% da capacidade do modelo ($M = 1.0$):
```bash
python scaling_laws.py --results_csv "training_results.csv" --target_N 8500 --target_M 1.0
```
O script gerará a predição e salvará uma representação 3D das superfícies de generalização estimadas em `scaling_law_surface.png`.

### Validação Real
Se você já treinou o modelo consolidado e tem o CSV com a métrica real observada (ex: `validation_results.csv` contendo a coluna `test_error`), use a flag `--validate`:
```bash
python scaling_laws.py --validate --val_csv "validation_results.csv"
```
Isso calcula a diferença percentual absoluta entre as previsões dos modelos e os resultados reais do treino em Fase 1+2.
