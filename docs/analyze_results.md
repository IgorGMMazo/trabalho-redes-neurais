# Performance Scaling Law Analysis Pipeline

This utility provides a CPU-only regression tool to fit Scaling Laws (Logarithmic Regression) on training results, predicting neural network performance (Train Loss and Test Accuracy) on larger dataset scales, and measuring prediction errors.

---

## 📋 Theoretical Framework

A common phenomenon in Deep Learning is that performance metrics (such as Loss and Accuracy) follow predictable trends as the dataset volume increases. This script models these performance curves using a logarithmic regression function:

$$y = a \ln(x) + b$$

Where:
* $x$: The dataset scale percentage (e.g. $1.0, 2.0, 5.0, \dots, 100.0$).
* $y$: The performance metric (Train Loss or Test Accuracy).
* $a$: Scaling rate coefficient.
* $b$: Bias offset coefficient.

### Fitting Method:
By transforming the input domain $X \to \ln(X)$, we map the logarithmic model to a first-degree polynomial regression (least-squares fit), which can be solved algebraically in $O(M)$ time using NumPy's `polyfit` without loading deep learning runtimes.

---

## ⚙️ Pipeline Specifications

### 1. Data Aggregation & Cleanup
To filter out noise across different random weight initializations, the analyzer executes the following operations:
1. Filters the CSV for the **maximum epoch** of each subset/run combination (the converged model state).
2. Group-by aggregate by the dataset percentage $x$, taking the **mean** of the metrics across the 5 initialization seeds.

### 2. Sequential Validation Scheme
For each subset scale index $k$ (starting from $k \ge 2$, i.e. $5\%$ onwards):
1. Fits the logarithmic model coefficients $a$ and $b$ using data points of subsets $P_0 \dots P_{k-1}$.
2. Evaluates the model at $P_k$ to obtain $y_{\text{pred}}$.
3. Computes the Absolute Error:
   $$\text{AbsError} = |y_{\text{actual}} - y_{\text{pred}}|$$
4. Computes the Relative Error (%):
   $$\text{RelError} = \frac{|y_{\text{actual}} - y_{\text{pred}}|}{y_{\text{actual}}} \times 100$$
5. Prints the sequential comparison table.

### 3. Matplotlib Visualizer
Generates a side-by-side subplot visualization:
* **Left Subplot**: Train Loss scaling curve.
* **Right Subplot**: Test Accuracy scaling curve (clipped to standard $[0.0, 1.0]$ bounds).
* **Fitting Bases**: Circled markers highlight which points were utilized to fit the logarithmic prediction line.

---

## 🚀 Execution Guide

### Local Terminal (CLI):
Execute the script from your terminal:
```bash
python analyze_results.py --csv-file "training_results.csv" --fit-points 3
```

### CLI Command Options:
* `--csv-file`: Path to the input metrics file (default: `training_results.csv`).
* `--fit-points`: Count of initial subset points used to fit the prediction line plotted in the chart (default: `3` [uses 1%, 2%, and 5%]).
* `--output-plot`: Filename to save the output figure (default: `scaling_law_plot.png`).

---

## ☁️ Google Colab Integration

For cloud environments, open and execute [analyze_results_colab.ipynb](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/analyze_results_colab.ipynb) which runs the analysis inline and displays the plots directly in your browser.
