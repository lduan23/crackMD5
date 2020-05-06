#include <cstdlib>
#include <iostream>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "md5c.cu"

using namespace std;

#define WORD_SIZE 10
#define CHARSET_SIZE 100

#define CHARSET "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789"
#define CHARSET_LEN (sizeof(CHARSET) - 1)

#define WORD_LENGTH 7

#define BLOCKS 16384UL
#define THREADS 512UL
#define HASHES_PER_KERNEL 128UL

uint8_t word_len;

char word[WORD_SIZE];
char charset[CHARSET_SIZE];
char cracked[WORD_SIZE];

__device__ char d_charset[CHARSET_SIZE];
__device__ char d_cracked[WORD_SIZE];

__host__ __device__ bool nextWord(uint8_t *len, char *word, uint32_t step) {

    uint32_t pos = 0;
    uint32_t add = 0;

    while (step > 0 && pos < WORD_LENGTH) {
        
        if (pos >= *len && step > 0) {
            step--;
        }

        add = step + word[pos];
        word[pos] = add % CHARSET_LEN;
        step = add / CHARSET_LEN;
        pos++;
    }

    if (pos > *len) {
        *len = pos;
    }

    if (pos > WORD_LENGTH) {
        return false;
    }

    return true;
}


__global__ void crack(char *charsetWord, uint8_t wordLen, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4) {

    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * HASHES_PER_KERNEL;

    __shared__ char sharedCharset[CHARSET_SIZE];

    uint8_t t_WordLen;
    uint32_t t_hash1, t_hash2, t_hash3, t_hash4;
    char t_CharsetWord[WORD_SIZE];
    char t_word[WORD_SIZE];

    memcpy(t_CharsetWord, charsetWord, WORD_SIZE);
    memcpy(&t_WordLen, &wordLen, sizeof(uint8_t));
    memcpy(sharedCharset, d_charset, sizeof(uint8_t) * CHARSET_SIZE);

    nextWord(&t_WordLen, t_CharsetWord, idx);

    for (uint32_t i = 0; i < HASHES_PER_KERNEL; i++) {

        for (uint32_t j = 0; j < t_WordLen; j++) {
            t_word[j] = sharedCharset[t_CharsetWord[j]];
        }

        getHash((unsigned char*)t_word, t_WordLen, &t_hash1, &t_hash2, &t_hash3, &t_hash4);

        if (t_hash1 == hash1 && t_hash2 == hash2 && t_hash3 == hash3 && t_hash4 == hash4) {
            memcpy(d_cracked, t_word, t_WordLen);
        }

        if (!nextWord(&t_WordLen, t_CharsetWord, 1)) {
            break;
        }
    }
}


int main (int argc, char **argv) {

    if (argc != 2 || strlen(argv[1]) != 32) {
        cout << argv[0] << " + target hash" << endl;
        return -1;
    }

    uint32_t hash[4];
    for (unsigned int i=0; i < 4; i++) {
        char tmp[16];
        strncpy(tmp, argv[1] + i*8, 8);
        sscanf(tmp, "%x", &hash[i]);
        hash[i] = (hash[i] & 0xFF000000) >> 24 | (hash[i] & 0x00FF0000) >> 8 | (hash[i] & 0x0000FF00) << 8 | (hash[i] & 0x000000FF) << 24;
    }

    memset(word, 0, WORD_SIZE);
    memset(cracked, 0, WORD_SIZE);
    memcpy(charset, CHARSET, CHARSET_LEN);

    word_len = WORD_LENGTH;

    cudaMemcpyToSymbol(d_charset, charset, sizeof(uint8_t) * CHARSET_SIZE, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_cracked, cracked, sizeof(uint8_t) * WORD_SIZE, 0, cudaMemcpyHostToDevice);
    
    char **words = new char*[1];
    cudaMalloc((void**)&words[0], sizeof(uint8_t) * WORD_SIZE);
    
    cudaEvent_t start, end;
    float ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    
    while (true) {

        cout << "Calculating..." << endl;

        cudaMemcpy(words[0], word, sizeof(uint8_t) * WORD_SIZE, cudaMemcpyHostToDevice);
        crack<<<BLOCKS, THREADS>>>(words[0], word_len, hash[0], hash[1], hash[2], hash[3]);
        bool next = nextWord(&word_len, word, THREADS * BLOCKS * HASHES_PER_KERNEL);

        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(cracked, d_cracked, sizeof(uint8_t) * WORD_SIZE, 0, cudaMemcpyDeviceToHost);

        bool found = (*cracked != 0);

        if (found) {
            cout << "Passcode is: " << cracked << endl;
            break;
        }

        if (!next) {
            cout << "No in range passcode found. Character range is A-Z, a-z, 0-9." << endl;
            break;
        }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cout << "Compute time = " << ms << "ms" << endl;

    cudaFree((void**)words[0]);
    delete[] words;
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}