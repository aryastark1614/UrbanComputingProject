Urban Computing 2024 course Leiden University (group 14)

All experiments are run on Python `3.9.21` setup via anaconda.
GPU used: GeForce GTX 1060 on CUDA+11.8, for CUDA support
we point to the regular torch installation guide. Our used
environment can be replicated using the `requirements.txt` assuming the same CUDA version is used.

Baseline setting is ran with the `baseline.py` code, 
Uni- and multivariate using the `regular.py` code, and 
the spatially weighting experiments with `weighted.py`.
Due to computational restrictions, we divide the evaluation from the training procedure
as performing this after training in the same procedure bottles the PC. 
The evaluation of uni- and multivariate are performed through `regular-eval.py`, similarly
the evaluation of the weighted experiments are can be ran through `weighted-eval.py`.

All files make use of the `data.py` code which utilizes the downloading of the data
(only performed once) and loading of the data into pandas dataframes.