# RBI_TrabajoFinal

Se quiere optar a la mejora del trabajo.

Final project of "Razonamiento Bajo Incertidumbre" from AI Master in UA (Universidad de Alicante). It consists in the analisys and commentation of the project Yang Liu, Xiang Ao, Zidi Qin, Jianfeng Chi, Jinghua Feng, Hao Yang, and Qing He. 2021. Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection. The uncertainty in the inference of the proposed neural network is done using different methods such as Bayesian Bootstrap, entropy and confidence intervals. 

## Execution

The executable file *uncertainty.py* from this repository performs the uncertainty estimation from the PC-GNN arquitecture. This file must be placed in the *main* folder of the original repository, alongsied *main.py* and executed with:

```sh
python3 uncertainty.py --config ./config/pcgnn_yelpchi.yml
```
