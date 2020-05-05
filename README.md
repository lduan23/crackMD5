# crackMD5
Parallel Computing of MD5 Hash Function

# How to compile
nvcc main.cu md5c.cu -Xcompiler -Wall -o main.out

nvcc main.cu md5c.cu -Xcompiler -Wall -Xcompiler -O3 -Xptxas -O3 -o main_o3.out