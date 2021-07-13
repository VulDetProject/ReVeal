#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    char *x = argv[1];
    if (atoi(x) > 3) {
        printf("Hello, %s!\n", x);
    }
    else {
        printf("Salutations, %s!\n", x);
    }
    printf("Goodbye, %s!\n", x);
}