# Political Polarization and Fractionalisation from Rational Values-Based Inference in an Agent-Based Graph Network

## Nicolas Navarre, Julie Pedersen, Adam Moore

<img src="data/normalized_data/analysis_files/figure-html/time plots-1.png" alt="time-plots" height="500">

<img src="data/normalized_data/analysis_files/figure-html/conservative-cluster size correlation-1.png" alt="cluster-plots" height="500">


## Data

`data` folder contains all of the data and corresponding plots, stats and analyses for the following simulations.
The full simulation data can be found in the public OSF: 

[All simulation data](https://osf.io/a249r/?view_only=fce8178a315041b381a4c52e6ea35fc2)

1. Normalized data (paper results)
    - 50 simulations
    - n_agents ~ N(100,15)
    - 1501 steps
2. Unnormalized Data
    - 100 simulations
    - n_agents ~ N(100,15)
    - 1501 steps
3. Example data (small scale simulation)
    - 10 simulations
    - n_agents ~ N(15,5)
    - 1001 steps

## Running simulations

To run simulations you need to run the following command in the base directory:

```
python model/save_simulation.py
```

This will save all simulations to  `./data/example_data/simulation_data/`.
You can edit this along with other simulation parameters in `model/save_simulation.py`.


## Analyses

To create the necessary data to produce the same figures and tables in the paper you can run the following script in the base directory:

```
python analyses/analyze_data.py
```

To render the figures, tables and statistics, you can render the analysis with the `analyses/analysis.qmd` script.
Using quarto, you can render the figures:

```
quarto render ./analyses/analysis.qmd --output-dir ../data/example_data/
```

