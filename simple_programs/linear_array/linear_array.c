#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>

#define PAGE_SIZE 4096ul 
#define TOTAL_NUM_ELEMENTS 65536ul
#define TOTAL_SIZE PAGE_SIZE*TOTAL_NUM_ELEMENTS

int main(void){
    /* 
        Linear array access (random page offset each page)
        Compile with -O0
    */


    srandom(time(0));   // Initialization, should only be called once.
    uint8_t* big_array = calloc(TOTAL_NUM_ELEMENTS,PAGE_SIZE*sizeof(uint8_t)); //uninitialized on purpose
    
    printf("Starting 0 initialization\n");

    for(size_t i = 0;i<TOTAL_SIZE;i+=1){
        big_array[i] = 0;
    }
    
    printf("Finished 0 initialization\n");

    printf("Starting linear array initialization\n");

    for(size_t i = 0;i<TOTAL_SIZE;i+=1){
        big_array[i] = random() % 255;
    }
    
    printf("Finished linear initialization\n");\

    printf("Starting linear scan with random offset\n");

    uint8_t final_byte = 0;
    for(size_t i = 0;i<TOTAL_NUM_ELEMENTS;i+=1){
        final_byte = big_array[i*PAGE_SIZE+(random()%PAGE_SIZE)];
    }
    printf("Final byte is %d\n",(int)final_byte);
    free(big_array);
    
    return 0;
}
