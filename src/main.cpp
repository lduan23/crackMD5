#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string.h>

#include <ctime>
#include <chrono>
#include <ratio>
#include <cmath>

#include "global.h"
#include "md5.h"

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

/* Length of test block, number of test blocks. */
#define TEST_BLOCK_LEN 1000
#define TEST_BLOCK_COUNT 1000

void MDString (char *);
void MDPrint (unsigned char [16]);

int main (int argc, char **argv) {

    char *str;

    if (argc > 1) {
        if (argv[1][0] == '-' && argv[1][1] == 's') {
            str = argv[2];
        }
    } else {
        cout << "Enter a string to run MD5 hashing >> ";
        string s;
        getline(cin, s);
        str = &s[0];
    }

    MDString(str);

    return 0;
}

void MDString (char *str) {
    MD5_CTX context;
    unsigned char digest[16];
    unsigned int len = strlen(str);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> ms;

    start = high_resolution_clock::now();
    MD5Init(&context);
    MD5Update(&context, (POINTER)str, len);
    MD5Final(digest, &context);
    end = high_resolution_clock::now();
    ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout << "MD5 of '" << str << "' = ";
    MDPrint(digest);
    cout << endl;

    cout << "Time = " << ms.count() << " ms" << endl;
}

void MDPrint (unsigned char digest[16]) {
    unsigned int i;
    for (i=0; i < 16; i++) {
        // cout << setfill('0') << setw(2) << setbase(16) << digest[i];
        printf ("%02x", digest[i]);
    }
}