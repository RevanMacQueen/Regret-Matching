# Regret-Matching
An implementation of the [Regret-Matching](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00153?casa_token=SlfFATJyWPsAAAAA:Cce3j4Nas--ytJ9DbagxY4x8hUKuFElvL_kiBL_Z_T9Ymv_SAoKMKIK2jajSmxQWs9wFlf9SWyiRrQ) algorithm. 

You may install this repo as a python library by running 
```
pip install .
```
from the root directory.

## Files 
All the implemenation files are contained withint `rm/`. `regret-matching.py` contains two implementations of Regret-Matching: `RegretMatching()` implements the standard algorithm that minimizes external regret; the empirical distriubution of play will converge to a coarse correlated equilibrium. `InternalRegretMatching()` implements an algorithm that minimizes internal regret: the empirical distriubution of play will converge to a correlated equilibrium. 

`shapley_game.ipynb` shows how to run this library on a simple game. 
