# Shared memory bank conflict investigation for mma::foreach_ij

## usage
1. Change ldm params at line 7, 8 in main.cu
2. Change arch at line 3 in Makefile
3. Build
```
make
```

4. Run
```
./bank-conflict.test
```

5. Check the result
```
[<matrix_b,16,8,16,half,col_major>, layout = row, ldm = 64] ----------...
  0(000)   0(080)   0(100)   0(180)   ... // the bank and smem memory index of loading frag.x[0] = smem[index];
  0(040)   0(0c0)   0(140)   0(1c0)   ... // the bank and smem memory index of loading frag.x[1] = smem[index];
  0(200)   0(280)   0(300)   0(380)   ... // the bank and smem memory index of loading frag.x[2] = smem[index];
  0(240)   0(2c0)   0(340)   0(3c0)   ... // the bank and smem memory index of loading frag.x[3] = smem[index];
[bank_conflict: 3]:  4  4  4  4  4  4 ... // access counter of each bank when loading frag.x[0] = smem[index];
[bank_conflict: 3]:  4  4  4  4  4  4 ... // access counter of each bank when loading frag.x[1] = smem[index];
[bank_conflict: 3]:  4  4  4  4  4  4 ... // access counter of each bank when loading frag.x[2] = smem[index];
[bank_conflict: 3]:  4  4  4  4  4  4 ... // access counter of each bank when loading frag.x[3] = smem[index];
```
