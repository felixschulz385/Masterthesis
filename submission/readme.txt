Replication Material for
## The negative external effects of deforestation on downstream populations in Brazil ##
Master's Thesis in Data Science in Business and Economics
Felix Schulz

---

## 1. Overview

This directory contains the replication files for my thesis project. 
The files include data, code, and instructions necessary to replicate the results presented in the paper. The scripts provided allow for a replication of all regression output. 

The generation of figures and tables with descriptive statistics requires the processing of raw data that is available by request. For a full overview of how all scripts in the repository are used, see Figure A8 of the thesis. 

## 2. Code

### 2.1 Requirements

- Software: R 
- Packages: 
    - tidyverse: A collection of R packages for data manipulation, visualization, and more.
    - lfe: Used for estimating linear models with multiple group fixed effects.
    - texreg: Functions for creating tables of regression results for LaTeX, HTML, and Word.
    - splm: Tools for spatial panel data econometrics.
    - Matrix: Provides classes and methods for dense and sparse matrices.
    - spdep: Functions for spatial dependence: weighting schemes, statistics, and models.

### 2.2 Running the Code

1. Step 1: Set the working directory
2. Step 2: Run code/analysis/analysis.R