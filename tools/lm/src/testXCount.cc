/*
 * testXCount --
 *	tests for XCount
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 2005-2006 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testXCount.cc,v 1.5 2010/06/01 03:31:14 stolcke Exp $";
#endif

#include <stdlib.h>
#ifdef PRE_ISO_CXX
# include <iostream.h>
#else
# include <iostream>
using namespace std;
#endif

#include "XCount.h"

int main()
{
    XCount a(1), b(1);

    cerr << "sizeof(XCcount) = " << sizeof(a) << endl;

    unsigned x = a + 10;

    a += 1;

    a = a + a;

    cerr << "x = " << x << " a = " << a << " b = " << b << endl;

    a = 40000;
    for (unsigned i = 0; i < 40000; i ++) {
    	a += 1;
	b += (XCount)1;
    }

    cerr << "a = " << a << " b = " << b << endl;

    exit(0);
}

