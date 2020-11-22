#include <stdio.h>
#include <stdlib.h>

int main( int argc, char **argv ){
    float *pos;
    int num = atoi( argv[1] );
    int spike_data = atoi( argv[2] );

    pos = (float *)malloc(sizeof(float)*3*num);
    fprintf(stderr, "%p\n", pos);

    free(pos);

    return 0;
}
