/*
 * Fixes for broken IEEE floating point on the Alpha/G++ platform
 */

#ifdef __alpha

#include <math.h>

/*
double
log10(double x)
{
	return log(x)/M_LN10;
}
*/

int
is_minus_infinity(float x)
{
	return x == (-1.0/0);
}

#endif

