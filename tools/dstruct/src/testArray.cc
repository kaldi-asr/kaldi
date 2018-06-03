//
// Testi for Array datastructure
//
// Copyright (c) 1995-2010 SRI International.  All Rights Reserved.
//
// $Header: /home/srilm/CVS/srilm/dstruct/src/testArray.cc,v 1.14 2013/03/22 05:34:27 stolcke Exp $
//

#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif

#define ZERO_INITIALIZE
#include "Array.cc"

#define BASE 10

#define SIZE 20

void
printArray(unsigned *a, unsigned start, unsigned end)
{
    for (unsigned i = start; i < end; i++) {
	cout << "i = " << i << " myarray[i] = " << a[i] << endl;
    }
}

int
main()
{
    ZeroArray<unsigned int> myarray(BASE, SIZE-BASE+1);
    Array<char *> array2;
    Array<const char *> array3;
    Array<float> array4;
    Array<double> array5;
    Array<char> array6;
    unsigned i;

    // unitialized array values
    printArray(myarray, BASE, BASE + myarray.size());

    cout << "size = " << myarray.size() << endl;

    for (i = BASE+1; i <= SIZE; i++) {
	myarray[i] = i * i;
    }

    printArray(myarray, BASE, BASE + myarray.size());

    cout << "size = " << myarray.size() << endl;

    cout << myarray.data()[BASE+3] << endl;

    cout << "*** testing copy constructor ***\n";

    Array<unsigned int> myarray2(myarray);

    for (i = BASE; i < BASE + myarray.size(); i++) {
	cout << "i = " << i << " myarray2[i] = " << myarray2[i] << endl;
    }

    cout << "*** runtime-sized array ***\n";

    unsigned dsize = 10;
    makeArray(unsigned, darray, dsize);

    for (i = 0; i < dsize; i++) {
	darray[i] = i * i * i;
    }

    printArray(darray, 0, dsize);

    exit(0);
}
