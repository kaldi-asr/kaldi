#if !defined(lint) && defined(SCCSIDS)
static	char sccsid[] = "@(#)qsort.c 1.1 91/11/13 SMI"; /* from UCB 4.2 83/03/0
89 */
#endif

/*
 * qsort.c:
 * Our own version of the system qsort routine which is faster by an average
 * of 25%, with lows and highs of 10% and 50%.
 * The THRESHold below is the insertion sort threshold, and has been adjusted
 * for records of size 48 bytes.
 * The MTHREShold is where we stop finding a better median.
 */

#define		THRESH		4		/* threshold for insertion */
#define		MTHRESH		6		/* threshold for median */

static void qst(char *base, char *max, int (*qcmp)(void*, void*),
					int qsz, int thresh, int mthresh);

/*
 * qsort:
 * First, set up some global parameters for qst to share.  Then, quicksort
 * with qst(), and then a cleanup insertion sort ourselves.  Sound simple?
 * It's not...
 */
void
qsort(void *base, int n, int size, int (*compar)(void *, void *))
{
	register char c, *i, *j, *lo, *hi;
	char *min, *max;
	int thresh = size * THRESH;		/* THRESHold in chars */
	int mthresh = size * MTHRESH;		/* MTHRESHold in chars */

	if (n <= 1)
		return;
	max = (char *)base + n * size;
	if (n >= THRESH) {
		qst(base, max, compar, size, thresh, mthresh);
		hi = (char *)base + thresh;
	} else {
		hi = max;
	}
	/*
	 * First put smallest element, which must be in the first THRESH, in
	 * the first position as a sentinel.  This is done just by searching
	 * the first THRESH elements (or the first n if n < THRESH), finding
	 * the min, and swapping it into the first position.
	 */
	for (j = lo = base; (lo += size) < hi; )
		if (compar(j, lo) > 0)
			j = lo;
	if (j != base) {
		/* swap j into place */
		for (i = (char *)base, hi = (char *)base + size; i < hi; ) {
			c = *j;
			*j++ = *i;
			*i++ = c;
		}
	}
	/*
	 * With our sentinel in place, we now run the following hyper-fast
	 * insertion sort.  For each remaining element, min, from [1] to [n-1],
	 * set hi to the index of the element AFTER which this one goes.
	 * Then, do the standard insertion sort shift on a character at a time
	 * basis for each element in the frob.
	 */
	for (min = base; (hi = min += size) < max; ) {
		while (compar(hi -= size, min) > 0)
			/* void */;
		if ((hi += size) != min) {
			for (lo = min + size; --lo >= min; ) {
				c = *lo;
				for (i = j = lo; (j -= size) >= hi; i = j)
					*i = *j;
				*i = c;
			}
		}
	}
}

/*
 * qst:
 * Do a quicksort
 * First, find the median element, and put that one in the first place as the
 * discriminator.  (This "median" is just the median of the first, last and
 * middle elements).  (Using this median instead of the first element is a big
 * win).  Then, the usual partitioning/swapping, followed by moving the
 * discriminator into the right place.  Then, figure out the sizes of the two
 * partions, do the smaller one recursively and the larger one via a repeat of
 * this code.  Stopping when there are less than THRESH elements in a partitio
n
 * and cleaning up with an insertion sort (in our caller) is a huge win.
 * All data swaps are done in-line, which is space-losing but time-saving.
 * (And there are only three places where this is done).
 */

static void
qst(char *base, char *max, int (*qcmp)(void*, void*), int qsz, int thresh, int mthresh)
{
	register char c, *i, *j, *jj;
	register int ii;
	char *mid, *tmp;
	int lo, hi;

	/*
	 * At the top here, lo is the number of characters of elements in the
	 * current partition.  (Which should be max - base).
	 * Find the median of the first, last, and middle element and make
	 * that the middle element.  Set j to largest of first and middle.
	 * If max is larger than that guy, then it's that guy, else compare
	 * max with loser of first and take larger.  Things are set up to
	 * prefer the middle, then the first in case of ties.
	 */
	lo = max - base;		/* number of elements as chars */
	do	{
		mid = i = base + qsz * ((lo / qsz) >> 1);
		if (lo >= mthresh) {
			j = (qcmp((jj = base), i) > 0 ? jj : i);
			if (qcmp(j, (tmp = max - qsz)) > 0) {
				/* switch to first loser */
				j = (j == jj ? i : jj);
				if (qcmp(j, tmp) < 0)
					j = tmp;
			}
			if (j != i) {
				ii = qsz;
				do	{
					c = *i;
					*i++ = *j;
					*j++ = c;
				} while (--ii);
			}
		}
		/*
		 * Semi-standard quicksort partitioning/swapping
		 */
		for (i = base, j = max - qsz; ; ) {
			while (i < mid && qcmp(i, mid) <= 0)
				i += qsz;
			while (j > mid) {
				if (qcmp(mid, j) <= 0) {
					j -= qsz;
					continue;
				}
				tmp = i + qsz;	/* value of i after swap */
				if (i == mid) {
					/* j <-> mid, new mid is j */
					mid = jj = j;
				} else {
					/* i <-> j */
					jj = j;
					j -= qsz;
				}
				goto swap;
			}
			if (i == mid) {
				break;
			} else {
				/* i <-> mid, new mid is i */
				jj = mid;
				tmp = mid = i;	/* value of i after swap */
				j -= qsz;
			}
		swap:
			ii = qsz;
			do	{
				c = *i;
				*i++ = *jj;
				*jj++ = c;
			} while (--ii);
			i = tmp;
		}
		/*
		 * Look at sizes of the two partitions, do the smaller
		 * one first by recursion, then do the larger one by
		 * making sure lo is its size, base and max are update
		 * correctly, and branching back.  But only repeat
		 * (recursively or by branching) if the partition is
		 * of at least size THRESH.
		 */
		i = (j = mid) + qsz;
		if ((lo = j - base) <= (hi = max - i)) {
			if (lo >= thresh)
				qst(base, j, qcmp, qsz, thresh, mthresh);
			base = i;
			lo = hi;
		} else {
			if (hi >= thresh)
				qst(i, max, qcmp, qsz, thresh, mthresh);
			max = j;
		}
	} while (lo >= thresh);
}
