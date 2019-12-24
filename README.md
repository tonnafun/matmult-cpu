# Optimized Matrix Multiplication on CPU

## TODO

- blocking might need to consider:
    - cache size and associativity
    - TLB size and associativity
- Why does the performance go down for some matrix sizes? Possible reasons:
    - When you go down a column of a matrix, maybe you will hit the same cache index. -> Conflict misses!
    - Conflict misses when accessing elements in different matrices which have the same cache index.
    - Solution?

## References

- Tan, Guangming, et al. "Fast implementation of DGEMM on Fermi GPU." Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis. ACM, 2011.
- Kirk, David B., and W. Hwu Wen-Mei. Programming massively parallel processors: a hands-on approach. Morgan kaufmann, 2016.
    
