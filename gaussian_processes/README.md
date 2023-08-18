# `flowsim.labs.gaussian_processes` Module File

## Introduction

The `labs` directory (and consequently Python module) contains a series of scripts 
allowing to conveniently interact with the `flowsim.models.simulator` code. 

The `labs.gaussian_processes` module allows 
to predict future data points and their confidence values given previously observed values and theoretical values predicted by the compartmental model.

The general idea behind the approach is that the theoretical compartmental model only provides generic tendencies for the evolution of all concerned compartment populations. It cannot predict detailed local flows.

The Gaussian process models the correlation between actual observations and theoretic predictions and provides more precise and better fitting predictions based on these correlations.

Further reading :
* Wikipedia on [Gaussian Processes](https://en.wikipedia.org/wiki/Gaussian_process)

## Scripts

Please note scripts should be run using the `-m` flag, like this:
```
python3 -m labs.gaussian_processes.script
```
Running the scripts with `--help` will provide a short description of available execution parameters.

* ###`predict.py`: 
  The script tries to predict the population of given SIR+H model compartments for the next time steps as well as the incertainty associated to this prediction given the observed compartment populations at previous time steps.
 

  ... to be completed ...

  
  ####Execution Parameters for `predict.py`

  * `-p`, `--prior`: pathname to the _prior data_ point set in CSV format. The _prior data_ point set is the set of compartment populations as predicted by the theoretic compartment model.
    
    If not provided, default observations for the Nancy ICU unit from are used (covering Jan-Apr 2020).
  * `-i`, `--input`: a `CSV` file containing a set of observed data points in time. Storage format is `time, value` 
    where `time` is expressed in days after the start of the simulation (usually January 6, 2020), and `value` the 
    observed data value.
    
    If not provided, default `data_chu_rea` data points from `labs.defaults.get_default_params()` are used.
  * `-n`: number `num` number of data points to consider for training. By default, all data points are used. 
    If set, only the first `num` data points will be used.
  * `-t`: number of days to predict
  * `--silentplot`: compute curve plots but do not display obtained curves (but allow to save them)
  * `--beautify`: compute beautified version of curves
  * `--noplot`: do not compute nor plot any curves
  * `-s`, `--save`: save output do files, prefixing filenames with prefix if provided
  * `--path`: save output files to provided path
  
