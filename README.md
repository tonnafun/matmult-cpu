# matmult-cpu
Optimized Matrix Multiplication on CPU

# TODO:

- blocking might need to consider:
    - cache size and associativity
    - TLB size and associativity
    

How come for a certain matrix size the performance goes down?
- Possible reasons:
    - When you go down a column of a matrix, maybe you will hit the same cache index. → Conflict misses!
    - Conflict misses when accessing elements in different matrices which have the same cache index.
- Solution: ??
    
