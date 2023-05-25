# 2022 Yangtze River Basin Heatwave attribution
Python code and Jupyter notebooks support the manuscript "Relative contributions of large-scale atmospheric circulation dynamics and anthropogenic warming to the unprecedented 2022 Yangtze River Basin Heatwave" by Zeqin Huang et al.

![CCA_YRB](Figures/Fig4_2022_CCA_construct_Z500.png)

## What's included
### Notebook
All the analysis and visualization code are performed using Jupyter notebook in a python environment.
The analysis notebooks include the ensemble constructed circulation analogue (CCA) analysis for the 2022 YRB heatwave specifically and for the whole research period (during 1979~2022).
> Both geopotential height at 500 hPa pressure level (Z500) and sea level pressure (SLP) are utilized for the ensemble CCA analysis.
* [CCA_analysis_for_2022_YRB_HW_Z500.ipynb](Notebook/CCA_analysis_for_2022_YRB_HW_Z500.ipynb)
* [CCA_analysis_for_2022_YRB_HW_SLP.ipynb](Notebook/CCA_analysis_for_2022_YRB_HW_SLP.ipynb)
> Figs. 1\~5 an S1\~S6 are generated using *matplotlib* and *proplot*
* [Fig1_unprecedented_YRB_HW.ipynb](Notebook/Fig1_unprecedented_YRB_HW.ipynb)
* [Fig2_S2_YRB_HW_nonstationary_fitting.ipynb](Notebook/Fig2_S2_YRB_HW_nonstationary_fitting.ipynb)
* [Fig3_Large-scale_conditions_2022.ipynb](Notebook/Fig3_Large-scale_conditions_2022.ipynb)
* [Fig4_CCA_plot_for_2022_YRB_HW.ipynb](Notebook/Fig4_CCA_plot_for_2022_YRB_HW.ipynb)
* [Fig5_S4_CCA_plot_for_historical.ipynb](Notebook/Fig5_S4_CCA_plot_for_historical.ipynb)
* [FigS1_SAT_HWD_spatial_trends.ipynb](Notebook/FigS1_SAT_HWD_spatial_trends.ipynb)
* [FigS2_GEV_fitting_for_HWD.ipynb](Notebook/FigS2_GEV_fitting_for_HWD.ipynb)
* [FigS5_CCA_plot_for_2022_YRB_HW_SLP.ipynb](Notebook/FigS5_CCA_plot_for_2022_YRB_HW_SLP.ipynb)
* [FigS6_CCA_plot_for_historical_SLP.ipynb](Notebook/FigS6_CCA_plot_for_historical_SLP.ipynb)
### Scripts
The nonstationary generalized extreme value (GEV) is adapted from https://github.com/clairbarnes/wwa.
* [nonstationary_fitting.py](Scripts/nonstationary_fitting.py)

## Datasets
* ERA5: The ERA5 data is available from the European Centre for Medium-range Weather Forecasts (ECMWF, https://www.ecmwf.int).
* JRA55: The JRA55 data is available from the Japanese 55-year Reanalysis (https://climatedataguide.ucar.edu/climate-data/jra-55). 

## Thanks
* https://github.com/russellhz/extreme_heat_CCA.
* https://github.com/clairbarnes/wwa.







