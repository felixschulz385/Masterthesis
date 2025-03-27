# Master Thesis: Deforestation, Water Quality and Health Externalities in Brazil 

This repository contains the code and data used in the master thesis "Deforestation, Water Quality and Health Externalities in Brazil" by [Felix Schulz](https://felixschulz385.github.io/).

## 🧭 Introduction

This thesis examines the causal impact of deforestation on water quality and public health outcomes in Brazil. By integrating remote sensing, hydrological, environmental, and health datasets, the analysis traces how upstream deforestation contributes to soil erosion and contaminates downstream water sources—ultimately affecting mortality rates and healthcare utilization.

To do this, I construct a detailed geospatial river network model of Brazil and compute corresponding drainage polygons. Deforestation data is spatially aggregated within these drainage areas and propagated through the river system. The resulting dataset forms the basis for an econometric analysis of environmental health externalities.

![Drainage polygon processing](/output/figures/drainage_preprocessing_example.png)

## 🗂 Repository Structure

This repository includes:

- **📁 `/code/`** – Modular scripts for data processing, exploratory analysis, and econometric modeling (primarily in R and Python).
- **📁 `/output/`** – Figures, intermediate results, and visualizations used in the thesis.
- **📁 `/data/`** – Replication data is available on request.

![Graphic illustrating the project workflow](/output/figures/data_workflow.png)

## License

This master thesis is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
Full license details can be found at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.
