
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "BlockMalloc.h"

#ifdef NEED_RAND48
extern "C" {
    double drand48();
}
#endif

using namespace std;

#define NUM_ITERS	3
#define NUM_MALLOCS	10000000
#define MAX_SIZE	100

static	void *data[NUM_MALLOCS];
static unsigned sizes[NUM_MALLOCS];

int main()
{
    unsigned i, j;

    for (i = 0; i < NUM_ITERS; i ++) {

	for (j = 0; j < NUM_MALLOCS; j ++) {
		unsigned size = (unsigned)(drand48() * MAX_SIZE);

		sizes[j] = size;
		data[j] = BM_malloc(size);
		assert(data[j] != 0);

		memset(data[j], -1, size);
	}

	cerr << "*** iteration " << i << endl;
	BM_printstats();

	for (j = 0; j < NUM_MALLOCS; j ++) {
		BM_free(data[j], sizes[j]);
	}
    }
}

