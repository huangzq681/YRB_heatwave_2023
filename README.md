# 2022 Yangtze River Basin Heatwave attribution
Python code and Jupyter notebooks support the manuscript "Relative contributions of large-scale atmospheric circulation dynamics and anthropogenic warming to the unprecedented 2022 Yangtze River Basin Heatwave" by Zeqin Huang et al.

![CCA_YRB](Figures/Fig5_2022_CCA_construct.png)

## What's included
### Notebook
All the analysis and visualization code are performed using Jupyter notebook in a python environment.
The analysis notebooks include the ensemble constructed circulation analogue (CCA) analysis for the 2022 YRB heatwave specifically and for the whole research period (during 1979~2022).
> Both geopotential height at 500 hPa pressure level (Z500) and sea level pressure (SLP) are utilized for the ensemble CCA analysis.
* [CCA_analysis_for_2022_YRB_HW_Z500.ipynb](Notebook/CCA_analysis_for_2022_YRB_HW_Z500.ipynb)
* [CCA_analysis_for_2022_YRB_HW_SLP.ipynb](Notebook/CCA_analysis_for_2022_YRB_HW_SLP.ipynb)
> Figs. 1\~7 an S1\~S10 are generated using *matplotlib* and *proplot*
* [Fig1_unprecedented_YRB_HW.ipynb](Notebook/Fig1_unprecedented_YRB_HW.ipynb)
* [Fig2_wget_fire_weather_index.ipynb](Notebook/Fig2_wget_fire_weather_index.ipynb)
* [Fig3_YRB_HW_nonstationary_fitting.ipynb](Notebook/Fig3_YRB_HW_nonstationary_fitting.ipynb)
* [Fig4_Large-scale_conditions_2022.ipynb](Notebook/Fig4_Large-scale_conditions_2022.ipynb)
* [Fig5_CCA_plot_for_2022_YRB_HW.ipynb](Notebook/Fig5_CCA_plot_for_2022_YRB_HW.ipynb)
* [Fig6_CCA_plot_for_historical.ipynb](Notebook/Fig6_CCA_plot_for_historical.ipynb)
* [Fig7_CCA_plot_for_2022_YRB_HW_with_different_analogue_numbers.ipynb](Notebook/Fig7_CCA_plot_for_2022_YRB_HW_with_different_analogue_numbers.ipynb)
* [FigS2_unprecedented_YRB_HW_characteristics.ipynb](Notebook/FigS2_unprecedented_YRB_HW_characteristics.ipynb)
* [FigS3_SAT_HW_characteristics_spatial_trends.ipynb](Notebook/FigS3_SAT_HW_characteristics_spatial_trends.ipynb)
* [FigS5_wget_fire_weather_index_anomalies.ipynb](Notebook/FigS5_wget_fire_weather_index_anomalies.ipynb)
* [FigS6_YRB_HW_nonstationary_fitting.ipynb](Notebook/FigS6_YRB_HW_nonstationary_fitting.ipynb)
* [FigS7_compare_HRLT_ERA5.ipynb](Notebook/FigS7_compare_HRLT_ERA5.ipynb)
* [FigS8_CCA_plot_for_historical_variance.ipynb](Notebook/FigS8_CCA_plot_for_historical_variance.ipynb)
* [FigS9_CCA_plot_for_2022_YRB_HW_SLP.ipynb](Notebook/FigS9_CCA_plot_for_2022_YRB_HW_SLP.ipynb)
* [FigS10_CCA_plot_for_historical_SLP.ipynb](Notebook/FigS10_CCA_plot_for_historical_SLP.ipynb)
### Scripts
The script for nonstationary generalized extreme value (GEV) fitting is adapted from https://github.com/clairbarnes/wwa.
* [nonstationary_fitting.py](Scripts/nonstationary_fitting.py)
### Figures
ALL the supporting figures for the manuscript.
## Datasets
* ERA5: The ERA5 data is available from the European Centre for Medium-range Weather Forecasts (ECMWF, https://www.ecmwf.int).
* JRA55: The JRA55 data is available from the Japanese 55-year Reanalysis (https://climatedataguide.ucar.edu/climate-data/jra-55).
* HRLT: A high-resolution (1 day, 1 km) and long-term (1961–2019) gridded dataset for temperature and precipitation (HRLT) across China (https://doi.pangaea.de/10.1594/PANGAEA.941329)
* FWI dataset: The fire weather index data is available from the Global Fire WEather Database (https://data.giss.nasa.gov/impacts/gfwed/). 
## Thanks
* https://github.com/russellhz/extreme_heat_CCA.
* https://github.com/clairbarnes/wwa.







