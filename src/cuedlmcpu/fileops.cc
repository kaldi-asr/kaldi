#include "head.h"
#include "fileops.h"

float random(float min, float max)
{
    return rand()/(Real)RAND_MAX*(max-min)+min;
}
