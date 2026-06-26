# Pipeline de Treinamento da ResNet-18 Escalável

Este documento descreve os detalhes técnicos de implementação da arquitetura ResNet-18 construída do zero e seu respectivo pipeline de otimização em PyTorch.

---

## 🔬 Especificações Científicas e Técnicas

### 1. Arquitetura Escalável por Largura ($\alpha$)
A capacidade da rede é controlada pelo fator multiplicador de largura ($\alpha$), que redimensiona o número de canais em todas as convoluções das camadas e blocos básicos da ResNet-18:
- **$\alpha = 0.125$ (12.5%)**: Canais base $[8, 16, 32, 64]$
- **$\alpha = 0.25$ (25%)**: Canais base $[16, 32, 64, 128]$
- **$\alpha = 0.5$ (50%)**: Canais base $[32, 64, 128, 256]$
- **$\alpha = 1.0$ (100%)**: Canais base $[64, 128, 256, 512]$ (ResNet-18 padrão)

### 2. Inicialização de Pesos (Xavier Uniform)
Todas as convoluções e camadas lineares do modelo são inicializadas deterministicamente a partir da distribuição uniforme de Xavier (Glorot):
$$W \sim U\left(-\sqrt{\frac{6}{d_{in} + d_{out}}}, \sqrt{\frac{6}{d_{in} + d_{out}}}\right)$$
Isso garante a manutenção da variância dos gradientes entre as camadas nas fases iniciais do treinamento sem pesos pré-treinados.

### 3. LR Scheduler (Warmup Linear + Cosine Annealing)
- **Warmup**: Nos primeiros 5% dos steps totais de treinamento, a taxa de aprendizado cresce linearmente de 0 até o valor máximo estabelecido ($2 \times 10^{-3}$).
- **Cosine Annealing**: Nos 95% de steps restantes, a taxa de aprendizado decai de forma cossoidal até 0:
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\pi \frac{T_{cur}}{T_{max}}\right)\right)$$

### 4. Orçamento Fixo de Treinamento e Otimização
- **Otimizador**: AdamW com decaimento de pesos de $1 \times 10^{-4}$ para regularização L2.
- **Orçamento por Steps**: Treinamento delimitado exclusivamente por número total de steps, garantindo compatibilidade uniforme em datasets de tamanhos distintos.
- **Cálculo de Inclinação de Perda (`loss_slope_final`)**: É calculada a inclinação da reta de regressão linear sobre o histórico de loss obtido nas últimas 10% atualizações de gradiente.

---

## 💻 Argumentos e Execução CLI (`train.py`)

```bash
python train.py \
    --train_dir "dataset/processed/train_fase_1" \
    --test_dir "dataset/processed/global_test" \
    --alpha 1.0 \
    --steps 10000 \
    --batch_size 64 \
    --lr 0.002 \
    --seeds 42 43 44 45 46 \
    --output_csv "training_results.csv"
```

O script salvará no final um arquivo CSV contendo os dados consolidados das 5 seeds executadas consecutivamente.
