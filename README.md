# naive_kernel() Performance 
Note: Used global memory only, NOT Consolidated read global memory, BUT write to global memory is Consolidated

```
width = 10000, height = 10000
CPU time = 1.033874e+03 ms
Allocate 3.814697e+02 MB on GPU
Performance = 2.22551 GFlop/s, kernel time = 8.986690e+01 ms
Checking computed result for correctness:
 Result = PASS

 ```

 # shared_kernel() Performance
 Note1: write to d_B of global memory is coalesced. BUT It's NOT coalesced while readding d_A from global memory

 NOTE2: It's important for a better performance to use 2-D block size rather than 1-D

 ```
width = 10000, height = 10000
CPU time = 1.066081e+03 ms
Allocate 3.814697e+02 MB on GPU
Performance = 4.36903 GFlop/s, Kernel time = 4.577675e+01 ms, 16 Threads Per Block
Checking computed result for correctness:
 Result = PASS

 ```


# faster_kernel() Performance
Note: the only and most important thing I did in this kernel is: 
make sure <b>coalesced</b> access d_A and d_B between global memory and shared memory.

that means, both <b>read</b> and <b>write</b> is coalesced.

we can see that the performance is around 65G Float elments per second!!! that's cool!

  ```
width = 10000, height = 10000
CPU time = 1.055775e+03 ms
Allocate 3.814697e+02 MB on GPU
Performance = 65.54411 GFlop/s, Kernel time = 3.051380e+00 ms, SpeedUP=345.999286
Checking computed result for correctness:
 Result = PASS
 ```
```
width = 10000, height = 20000
CPU time = 2.805697e+03 ms
Allocate 7.629395e+02 MB on GPU
Performance = 65.87951 GFlop/s, Kernel time = 6.071690e+00 ms, SpeedUP=462.094903
Checking computed result for correctness:
 Result = PASS
```
 NOTE: It's interesting that when I try to avoid bank conflict by request shared memory by padding method:
 ```
  __shared__ DATA_TYPE s_d[TILE_WIDTH][TILE_WIDTH+1];
 ```

 the performance would become worse, I don't understand why:

 ```
width = 10000, height = 10000
CPU time = 1.056310e+03 ms
Allocate 3.814697e+02 MB on GPU
Performance = 27.58545 GFlop/s, Kernel time = 7.250200e+00 ms, SpeedUP=145.693912
Checking computed result for correctness:
 Result = PASS
 ```