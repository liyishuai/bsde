# Option pricing with BSDE and Monte Carlo method

## Introdution
This project is to implement [Parallel Option Pricing with BSDE Method on GPU] and analyze its performance. More mathematical background can be found in [Backward Stochastic Differential Equations in Finance].

## How to use

### Build
    cmake .
    make all

### Execute
* `test_*.sh` runs a single implementation
* `make test` runs all four implementations

[Parallel Option Pricing with BSDE Method on GPU]:https://doi.org/10.1109/GCC.2010.47
[Backward Stochastic Differential Equations in Finance]:https://doi.org/10.1111/1467-9965.00022
