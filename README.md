# Amount of carbon fixed, transit time and fate of harvested wood products define the climate change mitigation potential of boreal forest management - A model analysis
  … by Holger Metzler, Samuli Launiainen, and Giulia Vico

<img src="https://github.com/goujou/BFCPM/blob/main/docs/source/_static/total_model_v2.png" width=400>

## Abstract
Boreal forests are often managed to maximize wood production, but other goals, among which climate change mitigation, are increasingly important. Examining synergies and trade-offs between boreal forest productivity and its potential for carbon sequestration and climate change mitigation in forest stands requires explicitly accounting for how long forest ecosystems and wood products retain carbon from the atmosphere (i.e., the carbon transit time). We propose a novel mass-balanced process-based compartmental model that allows following the carbon path from its photosynthetical fixation until its return to the atmosphere by autotrophic or heterotrophic respiration, or by being burnt as wood product. We investigate four management scenarios: mixed-aged pine, even-aged pine, even-aged spruce, even-aged mixed (pine and spruce) forest.
The even-aged clear-cut based scenarios reduced carbon amount in the system by one third in the first 18 yr.
Considering only the amount of carbon stored in the ecosystem, these initial losses are compensated after 42-45 yr. At the end of a 80-yr rotation, the even-aged forests hold up to 31\% more carbon than the the mixed-aged forest.
However, mixed-aged forest management is superior to even-aged forest management during almost the entire rotation when factoring in the carbon retention time away from the atmosphere, i.e., in terms of climate change mitigation potential. Importantly, scenarios that maximize productivity or amount of carbon stored in the ecosystems are not necessarily the most beneficial for carbon retention away from the atmosphere. These results underline the importance of considering carbon transit time when evaluating forest management options for potential climate change mitigation and hence explicitly tracking carbon in the system, e.g. via models like the one developed here. 

## Reproducing the manuscript results

**Reamrk:** This has been tested for WSL2 on Windows10.

First, create a new conda environment:

```
conda create --name BFCPM python=3.9
conda activate BFCPM
```

Then, it is necessary to install [bcg_md2](https://github.com/MPIBGC-TEE/bgc_md2).
This package stores the sub-models for the MeanTree, soil and the wood products and allows the computation of transit times of the implemented model.

### Installation of `bgc_md2`

```
git clone --recurse-submodules https://github.com/MPIBGC-TEE/bgc_md2.git
cd bgc_md2
./install_developer_conda.sh
cd ..

```

### Install `BFCPM`

```
git clone https://github.com/goujou/BFCPM.git
cd BFCPM
./install_developer_conda.sh
```

### After installation

All the data and figures for the manuscript have can be reproduced by two notebooks:
- notebooks/run_sims_for_paper_serial.ipynb
  - reproduces all the simulation data, can take hours to days
    - **NOTE:** If the variables `pre_spinup_date` and `sim_data` are not changed, then pre-computed (and provided) data as presented in the manuscript will be overwritten.
- notebooks/figures_notebook.ipynb
  - reproduces the figures from simulation data
  - pre-computed simulation data can be found in
    - data/pre-spinups/2023-07-25/: Pre-spinup for all simulations
    - data/simulations/2023-07-26/


[Model documentation](https://goujou.github.io/BFCPM/)

