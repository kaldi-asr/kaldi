/*
 * strtolplusb.cc --
 *	John Henderson's extension to strtol with 0b prefix.
 *
 * Jeff Bilmes <bilmes@ee.washington.edu>
 */

#include <stdlib.h>
#include <ctype.h>

long
strtolplusb(const char *nptr, char **endptr, int base)
{
  const char *i;
  long sign;

  /* We should only try to be clever if the base 2 is specified, or no
     base is specified, just like in the 0x case. */
  if (base != 2 && base != 0)
    return strtol(nptr,endptr,base);

  i = nptr;

  /* skip white space */
  while (*i && isspace(*i))
    i++;

  /* decide what the sign should be */
  sign = 1;
  if (*i) {
    if (*i == '+'){
      i++;
    } else if (*i == '-'){
      sign = -1;
      i++;
    }
  }

  /* If we're not at the end, and we're "0b" or "0B", then return the result
     base 2.  Let strtol do the work for us. */
  if (*i && *i == '0' && *(i+1) && (*(i+1) == 'b' || *(i+1)=='B') 
      && *(i+2) && (*(i+2)=='1' || *(i+2)=='0'))
    return sign*strtol(i+2,endptr,2);
  else
    /* otherwise use the given base */
    return strtol(nptr,endptr,base);
}

