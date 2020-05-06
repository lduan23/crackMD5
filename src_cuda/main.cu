#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>

#include <ctime>
#include <chrono>
#include <ratio>
#include <cmath>

// #include "global.h"
// #include "md5.h"

#include "md5c.cu"

using namespace std;
// using std::chrono::high_resolution_clock;
// using std::chrono::duration;

/* Length of test block, number of test blocks. */
// #define TEST_BLOCK_LEN 1000
// #define TEST_BLOCK_COUNT 1000

#define WORD_LIMIT 10
#define CHARSET_LIMIT 100

#define CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CHARSET_LEN (sizeof(CHARSET) - 1)

#define WORD_LEN_MIN 7
#define WORD_LEN_MAX 7

#define TOTAL_BLOCKS 16384UL
#define TOTAL_THREADS 512UL
#define HASHES_PER_KERNEL 128UL


uint8_t g_wordLength;

char g_word[WORD_LIMIT];
char g_charset[CHARSET_LIMIT];
char g_cracked[WORD_LIMIT];

__device__ char g_deviceCharset[CHARSET_LIMIT];
__device__ char g_deviceCracked[WORD_LIMIT];


__device__ __host__ bool nextWord(uint8_t *length, char *word, uint32_t increment) {

    uint32_t idx = 0;
    uint32_t add = 0;

    while (increment > 0 && idx < WORD_LIMIT) {
        
        if (idx >= *length && increment > 0) {
            increment--;
        }

        add = increment + word[idx];
        word[idx] = add % CHARSET_LEN;
        increment = add / CHARSET_LEN;
        idx++;
    }

    if (idx > *length) {
        *length = idx;
    }

    if (idx > WORD_LEN_MAX) {
        return false;
    }

    return true;
}


__global__ void md5Crack(uint8_t wordLength, char *charsetWord, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4) {

    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * HASHES_PER_KERNEL;

    __shared__ char sharedCharset[CHARSET_LIMIT];

    char t_CharsetWord[WORD_LIMIT];
    char t_TextWord[WORD_LIMIT];
    uint8_t t_WordLength;
    uint32_t t_hash1, t_hash2, t_hash3, t_hash4;

    memcpy(t_CharsetWord, charsetWord, WORD_LIMIT);
    memcpy(&t_WordLength, &wordLength, sizeof(uint8_t));
    memcpy(sharedCharset, g_deviceCharset, sizeof(uint8_t) * CHARSET_LIMIT);

    nextWord(&t_WordLength, t_CharsetWord, idx);

    for (uint32_t hash = 0; hash < HASHES_PER_KERNEL; hash++) {

        for (uint32_t i = 0; i < t_WordLength; i++) {
            t_TextWord[i] = sharedCharset[t_CharsetWord[i]];
        }

        md5Hash((unsigned char*)t_TextWord, t_WordLength, &t_hash1, &t_hash2, &t_hash3, &t_hash4);

        if (t_hash1 == hash1 && t_hash2 == hash2 && t_hash3 == hash3 && t_hash4 == hash4) {
            memcpy(g_deviceCracked, t_TextWord, t_WordLength);
        }

        if (!nextWord(&t_WordLength, t_CharsetWord, 1)) {
            break;
        }
    }
}



/*
void MDString (char *);
void MDPrint (unsigned char [16]);
*/

int main (int argc, char **argv) {

    // Parse argument
    if (argc != 2 || strlen(argv[1]) != 32) {
        cout << argv[0] << " + target hash" << endl;
        return -1;
    }

    // Convert hash to 4 u32 integers
    uint32_t hash[4];

    for (unsigned int i=0; i < 4; i++) {
        char tmp[16];

        strncpy(tmp, argv[1] + i*8, 8);
        sscanf(tmp, "%x", &hash[i]);
        hash[i] = (hash[i] & 0xFF000000) >> 24 | (hash[i] & 0x00FF0000) >> 8 | (hash[i] & 0x0000FF00) << 8 | (hash[i] & 0x000000FF) << 24;
    }

    memset(g_word, 0, WORD_LIMIT);
    memset(g_cracked, 0, WORD_LIMIT);
    memcpy(g_charset, CHARSET, CHARSET_LEN);

    g_wordLength = WORD_LEN_MIN;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    char **words = new char*[1];

    cudaMemcpyToSymbol(g_deviceCharset, g_charset, sizeof(uint8_t) * CHARSET_LIMIT, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_deviceCracked, g_cracked, sizeof(uint8_t) * WORD_LIMIT, 0, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&words[0], sizeof(uint8_t) * WORD_LIMIT);

    while (true) {

        bool next = false;
        bool found = false;

        cudaMemcpy(words[0], g_word, sizeof(uint8_t) * WORD_LIMIT, cudaMemcpyHostToDevice);

        md5Crack<<<TOTAL_BLOCKS, TOTAL_THREADS>>>(g_wordLength, words[0], hash[0], hash[1], hash[2], hash[3]);

        next = nextWord(&g_wordLength, g_word, TOTAL_THREADS * TOTAL_BLOCKS * HASHES_PER_KERNEL);

        char word[WORD_LIMIT];

        for (int i=0; i < g_wordLength; i++) {
            word[i] = g_charset[g_word[i]];
        }

        cout << "Current at: " << string(word, g_wordLength) << " (" << (uint32_t)g_wordLength << ")" << endl;

        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(g_cracked, g_deviceCracked, sizeof(uint8_t) * WORD_LIMIT, 0, cudaMemcpyDeviceToHost);

        if (found = *g_cracked != 0) {
            cout << "Cracked: " << g_cracked << endl;
            break;
        }

        if (!next || found) {
            if (!next && !found) {
                cout << "Find nothing! " << endl;
            }

            break;
        }
    }

    cudaFree((void**)words[0]);

    delete[] words;

    float ms = 0;
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);

    cout << "Compute time = " << ms << "ms" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // char *str;

    // if (argc > 1) {
    //     if (argv[1][0] == '-' && argv[1][1] == 's') {
    //         str = argv[2];
    //     }
    // } else {
    //     cout << "Enter a string to run MD5 hashing >> ";
    //     string s;
    //     getline(cin, s);
    //     str = &s[0];
    // }

    // MDString(str);

    return 0;
}

/*
void MDString (char *str) {
    MD5_CTX context;
    unsigned char digest[16];
    unsigned int len = strlen(str);

    high_resolution_clock::time_point start, start_all;
    high_resolution_clock::time_point end, end_all;
    duration<double, std::milli> ms;

    start_all = high_resolution_clock::now();
    start = high_resolution_clock::now();
    MD5Init(&context);
    end = high_resolution_clock::now();
    ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << "MD5Init runtime = " << ms.count()*1000 << " us" << endl;

    start = high_resolution_clock::now();
    MD5Update(&context, (POINTER)str, len);
    end = high_resolution_clock::now();
    ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << "MD5Update runtime = " << ms.count()*1000 << " us" << endl;
    
    start = high_resolution_clock::now();
    MD5Final(digest, &context);
    end = high_resolution_clock::now();
    ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << "MD5Final runtime = " << ms.count()*1000 << " us" << endl;

    end_all = high_resolution_clock::now();
    ms = std::chrono::duration_cast<duration<double, std::milli>>(end_all - start_all);

    cout << "\nMD5 of '" << str << "' = ";
    MDPrint(digest);
    cout << endl;

    cout << "Total runtime = " << ms.count()*1000 << " us" << endl;
}

void MDPrint (unsigned char digest[16]) {
    unsigned int i;
    for (i=0; i < 16; i++) {
        printf ("%02x", digest[i]);
    }
}

*/