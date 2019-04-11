# Power spectrum code

Parallelized python code to compute the mode-coupling integrals and hence the real-space
power spectrum of biased tracers from Convolution Lagrangian Effective Field
Theory, as described in:

Z.Vlah, E.Castorina, M.White

The Gaussian streaming model and Convolution Lagrangian effective field theory

JCAP 12(2016)007, [https://arxiv.org/abs/1609.02908]

The code is parallelized using 'pool' object in python. 

## Requirements
This code requires NumPy and SciPy and makes use of the "multiplicative convolutional fast integral transforms" library:

https://github.com/eelregit/mcfit

which you will need to install.

## Code setup

The main piece of code is in cleftpool.py, which has CLEFT class to create PT kernels and make_table function
in the same file uses these kernels to create a table of P(k) where different columns correspond
to contribution of different bias parameters.


We provide a script "main.py" which can be directly run as follows: 
```python
python main.py \-\-pfile path_to_linear_ps_file
```

- Other arguments that can be provided are can be seen by calling: 
```python
python main.py \-\-help
```

- Some of the important arguments are:
  - pfile: path to the file containing linear power spectrum at z=0    
  - npool: number of processes to spawn, default=32
  - z: redshift, default = 0 
  - M: Omega-Matter
  - nk: Number of log-spaced k-values to evaluate power spectrum at, default=50
  - kmin: the minimum k value, default=0.001
  - kmax: the maximum k value, default=3
  - k, p: if 'pfile' is not specified, one can altternatively pass in the k, p numpy arrays for linear power spectrum<br>
  Either of (k, p) or (pfile) argument is required to be passed in

Other parameters such as --qfile and --rfile can be specified to point to the save kernels in order to speed up the code when computing power spectrum for same cosmology.

- Basic syntax to call it in jupyter notebooks or as a part of other code is: 
```python
import cleftpool as cpool
cl = cpool.CLEFT(pfile = pfile,  npool=32)
pk = cpool.make_table(cl, kmin = 0.002, kmax = 1, nk = 200, npool=32, z = 1, M = 0.3)
```

- Timing: With 32 cores on a single node of Cori-jupyter hub, it takes ~35 seconds to
compute power spectra at 200 k-values.

<aside class="notice">
The code uses package *mcfit* to do bessel integrals in the file qfuncpool.py. This is handled by function *dosph* in class Qfuck and it takes in the integration ranges 'q1' and 'q2'. While the current set-up for these values has been tested for many cosmologies, it has the potential of outputing 'nans' in some rare cases. This can be solved by changing the tilt or the said intergation ranges for the corresponding function. 
</aside>
