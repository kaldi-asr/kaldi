/*
 * testLattice --
 *	Test for Lattice class
 */

#ifndef lint
static char Copyright[] = "Copyright (c) 1997, SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lattice/src/testLattice.cc,v 1.5 2006/01/16 19:36:43 stolcke Exp $";
#endif

#include <stdio.h>

#include "Vocab.h"
#include "Lattice.h"

int
main (int argc, char *argv[])
{
    Vocab vocab;
    Lattice *lat = new Lattice(vocab);
    assert(lat != 0);

    File f(stdout);
    cerr << "Empty lattice:\n";
    lat->writePFSG(f);

    cerr << "Empty HTK lattice:\n";
    lat->writeHTK(f);

    if (argc == 2) {
	File file(argv[1], "r");

	lat->readPFSG(file);
    }

    cerr << "Read lattice:\n";
    lat->writePFSG(f);

    cerr << "Read lattice in HTK format:\n";
    lat->writeHTK(f);
    delete lat;
    lat = 0;

    exit(0);
}
