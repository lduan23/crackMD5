# crackMD5
Parallel Computing of MD5 Hash Function  


## How to compile
nvcc main.cu md5c.cu -Xcompiler -Wall -o main  
nvcc main.cu md5c.cu -Xcompiler -Wall -Xcompiler -O3 -Xptxas -O3 -o main_o3  


## How to run
./main_o3.out 8430894cfeb54a3625f18fe24fce272e  


## Sample output
$ ./main_o3 8430894cfeb54a3625f18fe24fce272e  
Calculating...  
Passcode is: AAAAAAA  
Compute time = 657.71ms  


## MD5 Test suite
MD5 of 'AAAAAAA' = 8430894cfeb54a3625f18fe24fce272e  
MD5 of 'AAAAAAa' = 9d776eec0059649aacaf127482f72c28  
