# stream-seepage
Python implementation of a closed-form approximation of steady-state seepage
from a clogged disconnected stream.

The closed-form approximation is detailed in the article "Seepage from disconnected streams: from asymptotic regimes to a closed-form approximation" (DOI:). It works for both transitionally and fully disconnected conditions, and is implemented for van Genuchten-Mualem and Brooks-Corey-Burdine unsaturation parametrizations.


### repository structure
Different seepage formulas are implemented in rate.py.

A comparison of the different seepage expressions is presented in the notebook
demo.ipynb, which also serves as a simple working example.

The unsaturated hydraulic conductivity functions are implemented in uhc.py.

Models 1 and 3 of Rosetta3 are implemented in rose.py to generate unsaturated properties from textural information, either class name or sand, silt and clay fractions.

Numerical results presented in the article were generated with the scripts and notebooks in the directory *experiments*. They are fully documented. The associated data files are stored in the directory *data*.

The file *data/munoz_unsat_ref.csv* contains the list of van Genuchten properties compiled by Muñoz-Vega 2025 (WRR) and used in Fig.10.

The file *data/streambeds.json* contains the list of streambed and aquifer properties reproted in the literature and used in Fig.6. 


### list of seepage formulas
The following expressions are implemented in rate.py:

- q_exact (full and transitional disconnection):
Numerical solution of the steady state Richard's equation for given boundary conditions, i.e., stream stage and water table depth.
- q_exact_full (full disconnection):
Numerical solution of the unsaturated Darcy equation across a clogging layer.
- q_approx (full and transitional disconnection):
Closed-form approximation constructed in the article.
- q_approx_full (full disconnection):
Closed-form approximation constructed in the article.
- q_modflow (full disconnection):
Approximate solution implemented in MODLFOW, obtained by setting the suction at the streambed-aquifer interface equal to zero.


### Quickstart

clone repository:
```
git clone https://github.com/jpaccolat/seepage.git
```

create conda environement:
```
conda create --name env python=3.14.3
conda activate env
conda install numpy=2.4.2 scipy=1.17.1 matplotlib=3.10.8 pandas=3.0.1
```
This is sufficient to evaluate the different seepage expression and test them in the notebook *demo.ipynb*. To run the experiments, the following packages are also required:

```
conda install tqdm=4.67.3
conda install conda-forge::alphashape=1.3.1
conda install conda-forge::threadpoolctl=3.6.0
pip install rosetta-soil==0.3.2 colorstamps==0.1.3
```

### How to cite