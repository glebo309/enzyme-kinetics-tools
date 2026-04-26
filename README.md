# Enzyme Kinetics Tools

Two tools for fitting enzyme kinetics data from substrate depletion time courses.

## Tools

### 1. Michaelis-Menten Fitter (`michaelis_menten_fitter.py`)

Simple Dash app for quick Michaelis-Menten fitting. Paste data from Excel, get Km and Vmax with confidence intervals.

- Nonlinear least-squares curve fitting
- Lineweaver-Burk, Eadie-Hofstee, and Hanes-Woolf linearizations
- Interactive Plotly plots
- Copy-paste from Excel or CSV

```bash
pip install dash plotly pandas numpy scipy
python michaelis_menten_fitter.py
# Open http://127.0.0.1:8050
```

### 2. Advanced Bayesian Enzyme Kinetics (`multimodel_enzyme_kinetics_app.py`)

Streamlit app for multi-model Bayesian fitting of substrate depletion curves using PyMC and ODE integration.

**Supported models:**
- Michaelis-Menten (sQSSA and tQSSA formulations)
- Hill equation (cooperative binding)
- Competitive, noncompetitive, and uncompetitive inhibition
- Substrate inhibition

**Features:**
- Global fitting across multiple substrate concentrations
- Bayesian parameter estimation with posterior distributions (PyMC/NUTS)
- Automatic model comparison (WAIC, LOO)
- ODE-based fitting (not initial-rate approximations)
- Residual analysis and diagnostic plots

```bash
pip install streamlit numpy pandas scipy plotly pymc pytensor arviz
streamlit run multimodel_enzyme_kinetics_app.py
```

## Requirements

- Python 3.10+
- See individual tool sections for dependencies
