# crackMD5
Parallel Computing of MD5 Hash Function

# How to compile
nvcc main.cu md5c.cu -Xcompiler -Wall -o main.out

nvcc main.cu md5c.cu -Xcompiler -Wall -Xcompiler -O3 -Xptxas -O3 -o main_o3.out


# Test suite
MD5 test suite:
MD5 ("") = d41d8cd98f00b204e9800998ecf8427e
MD5 ("a") = 0cc175b9c0f1b6a831c399e269772661
MD5 ("abc") = 900150983cd24fb0d6963f7d28e17f72
MD5 ("message digest") = f96b697d7cb7938d525a2f31aaf161d0
MD5 ("abcdefghijklmnopqrstuvwxyz") = c3fcd3d76192e4007dfb496cca67e13b
MD5 ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") =
d174ab98d277d9f5a5611c2c9f419d9f
MD5 ("123456789012345678901234567890123456789012345678901234567890123456
78901234567890") = 57edf4a22be3c955ac49da2e2107b67a


MD5 of '1234567' = fcea920f7412b5da7be0cf42b8c93759
MD5 of 'aaaaaaa' = 5d793fc5b00a2348c3fb9ab59e5ca98a
MD5 of 'aaaaaas' = b86539b0a61fdfdaef35a7c89edd0dc0