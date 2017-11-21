static char rcsid[] = "$Id: sentag.c,v 1.9 1996/08/13 15:57:35 robertm Rel $";
/*************************************************************
 * sentag.c
 *------------------------------------------------------------
 * Intended to do the best possible sentence tagging of
 * text data from journalistic sources.  Input format is
 * the typical TIPSTER-style SGML, in which the critical
 * tags required are indicated below, and other tags are
 * passed through without modifications:
 *
 *	<DOC id=artid-string>
 *	...
 *	<TEXT>
 *	<p>
 *	All text should be prepared with one paragraph on a line, regardless \
 *	how long it is (up to 65536 chars).
 *	<p>
 *	The sentag program will make changes within the "TEXT" region only. This \
 *	is an example.
 *	<p>
 *	In addition to putting one whole paragraph on one line, other cleaning up \
 *	may be needed so that output sentences are tidy. This might include removing \
 *	"datelines", etc.
 *	<p>
 *	Note that closing tags are implicit for paragraphs. The same will apply to \
 *	sentence tags in the output.
 *	</TEXT>
 *	...
 *	</DOC>
 *
 * Output format is:
 *
 *	<DOC id=artid-string>
 *	...
 *	<TEXT>
 *	<p id=artid-string.1>
 *	<s>
 *	All text should be prepared with one paragraph on a line, regardless \
 *	how long it is (up to 65536 chars).
 *	<p id=artid-string.2>
 *	<s>
 *	The sentag program will make changes within the "TEXT" region only.
 *	<s>
 *	This is an example.
 *	<p id=artid-string.3>
 *	<s>
 *	In addition to putting one whole paragraph on one line, other cleaning up \
 *	may be needed so that output sentences are tidy.
 *	<s>
 *	This might include removing "datelines", etc.
 *	<p id=artid-string.4>
 *	<s>
 *	Note that closing tags are implicit for paragraphs.
 *	<s>
 *	The same will apply to sentence tags in the output.
 *	</TEXT>
 *	...
 *	</DOC>
 *
 * In a nutshell, this program applies unique ID strings to all
 * paragraph tags, inserts an initial <s> tag at the start of each
 * paragraph, and for each period "." character that marks the end of
 * a sentence within a paragraph, it replaces the following space with
 * "\n<s>\n".
 *
 * This program operates as a pipeline filter.
 *
 * By default, it looks in "./addressforms" for a list of
 * sentence-internal abbreviations, and in "./sent-init.vocab" for a
 * list of words that would only be capitalized at the beginning of a
 * sentence.  The arguments "-a abbrevfile" and "-i sent-init.list"
 * can override the defaults.
 *
 * If either "abbrev" or "sent-init" file is not found, the program exits.
 *
 * A "sent-init.candidate" file is created, containing all the cases
 * in which a capitalized word following a period has been _assumed_
 * to be a continuation of an abbreviated proper noun phrase
 * (e.g. U.S. Treasury).  This "candidate" file (and a histogram of
 * its tokens) should be reviewed to look for (classes of) possible
 * missed boundaries.  Sentence breaks are NOT applied to these cases,
 * and a second pass over the same input data should be made if the
 * "sent-init" file is updated to include any of these candidates.
 * The argument "-t candidate.file" will override the default name.
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <search.h>

#define BUFSIZE 65536
#define MAXABRV 2048
#define MAXIVCB 1024
#define MAXBRKS 256
#define IDLEN   64
#define MAXSENTLEN 4096

char *abbrevs[MAXABRV];		/* contains sentence-internal abbrevs */
char idstr[IDLEN];
struct si_word {
    char *wd;
} si_node, s_init_wd[MAXIVCB];	/* contains non-capitalized words */

int n_abbrevs = 0;
int n_mid_abbrevs, n_s_init = 0, pid;

FILE *tfp;

/* --------------------------------------------------
 * w_compare() : comparison function for bsearch()
 */
int w_compare( w1, w2 )
  struct si_word *w1, *w2;
{
    return strcmp( w1->wd, w2->wd );
}


main( ac, av )
  int ac;
  char **av;
{
    FILE *afp, *ifp;
    int c, i, j, inText;
    char buf[BUFSIZE], *cp;
    extern int optind, opterr;
    extern char *optarg;
    int w_compare();

/* Handle options or defaults
 */
    afp = ifp = tfp = NULL;
    while (( c = getopt( ac, av, "a:i:t:" )) != -1 )
        switch ( c )
        {
	  case 'a':
	    if (( afp = fopen( optarg, "r" )) == NULL ) {
		fprintf( stderr, "Unable to open abbrev file %s\n", optarg );
		exit(1);
	    }
	    break;
	  case 'i':
	    if (( ifp = fopen( optarg, "r" )) == NULL ) {
		fprintf( stderr, "Sent-init.vocab file %s not found.\n", optarg );
		exit(1);
	    }
	    break;
	  case 't':
	    if (( tfp = fopen( optarg, "w" )) == NULL ) {
		fprintf( stderr, "Can't create %s -- quitting.\n", optarg );
		exit(1);
	    }
	    break;
	  default:
	    fprintf( stderr, "Usage: %s [-a abbrevs] [-i sent-init.vocab]\n", av[0] );
	    fprintf( stderr, "version: %s\n", rcsid );
	    exit(1);
	}

/* Always create a table of uncertain capitalized words
 */
    if ( ! tfp && ( tfp = fopen( "sent-init.candidate", "a" )) == NULL ) {
	fprintf( stderr, "Can't create/append-to ./sent-init.candidate\n" );
	exit(1);
    }

/* Load typical sentence-initial words (capitalized only when sentence-intial)
 * -- input list file must be presorted alphabetically
 */
    if ( ! ifp && ( ifp = fopen( "sent-init.vocab", "r" )) == NULL ) {
	fprintf( stderr, "File ./sent-init.vocab not found.\n" );
	exit(1);
    }
    while ( n_s_init < MAXIVCB && fgets( buf, BUFSIZE, ifp ) != NULL )
	if ( buf[0] != '#' )
	    s_init_wd[ n_s_init++ ].wd = strdup( strtok( buf, "\n" ));
    fclose( ifp );

/* Load definite within-sentence abbrevs
 */
    if ( ! afp && ( afp = fopen( "addressforms", "r" )) == NULL ) {
	fprintf( stderr, "Unable to open file ./addressforms\n" );
	exit(1);
    }
    while ( n_abbrevs < MAXABRV && fgets( buf, BUFSIZE, afp ) != NULL )
	if ( buf[0] != '#' )
	    abbrevs[ n_abbrevs++ ] = strdup( strtok( buf, "." ));
    fclose( afp );
    n_mid_abbrevs = n_abbrevs;

/* Add some special abbrevs to the list
 */
    abbrevs[ n_abbrevs++ ] = strdup( "Dr" );
    abbrevs[ n_abbrevs++ ] = strdup( "St" );

/* Scan and tag text data
 */
    inText = 0;
    *idstr = 0;
    while ( gets( buf ))
    {
	if (strlen(buf) > BUFSIZE)
	  {
	    fprintf( stderr, "input buffer size exceeded!!\n" );
	    fprintf( stderr, "last input:\n%s\n", buf );
	    exit(-1);
	  }
	if ( !inText ) {
	    if ( buf[0] == '<' )
		switch ( buf[1] )
		{
		  case 'D':
		    if ( !strncmp( buf, "<DOC id=", 8 )) {
			strcpy( idstr, &buf[8] );
			if (( cp = strchr( idstr, '>' )) != NULL )
			    *cp = 0;
			else
			  fprintf( stderr, "bad ID??\nid=%s\n", idstr );
		    }
		    break;
		  case 'T':
		    if ( !strncmp( buf, "<TEXT>", 6 )) {
			if ( ! *idstr ) {
			    fprintf( stderr, "No DOCID string -- quitting.\n" );
			    exit(1);
			}
			inText = 1;
			pid = 0;
		    }
		    break;
		  default:
		    break;
		}
	    puts( buf );
	}
	else {
	    if ( buf[0] == '<' )
		switch ( buf[1] )
		{
		  case 'p':
		    pid++;
		    printf( "<p id=%s.%d>\n", idstr, pid );
		    break;
		  case '/':
		    if ( !strncmp( buf, "</TEXT>", 7 ))
		      inText = 0;
		    puts( buf );
		    break;
		  default:
		    if (( !strncmp( buf, "<speaker>", 9 ))
		        || ( !strncmp( buf, "<comment>", 9 )))
		      {
			puts( buf );
		      }
		    else
		      {
			fprintf( stderr, "Warning: passing odd markup in %s:\n\t%s\n", idstr, buf );
			puts( buf );
		      }
		}
	    else {
		strcat( buf, " " );
		sentBreak( buf );
	    }
	}
    }
    exit(0);
}


char *ucs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
char *lcs = "abcdefghijklmnopqrstuvwxyz";
char *crp_abbrv[] = { "CORP", "INC", "CO", "PLC", "LTD", "BHD", "CIE",
		      "DEPT", "LTDA", "MFG", "SPA" };
int n_crp_abbrv = 11;
char *time_zone[] = { "EST", "EDT", "PST", "PDT", "CST", "CDT", "MST", "MDT", "GMT" };
int n_time_zone = 9;

#define MAXWDLEN 64
#define DoNextPeriod continue

sentBreak( buf )
  char *buf;
{
    char *period[MAXBRKS], *start, perchr, nxtwd[MAXWDLEN];
    char *nxtch, *nxtuc, *nxtsp, *prvch, *prvsp, *endwd, *prvwd, *endpg;
    char *openbracketp;
    int n_per, i, j, k;

    n_per = 0;
    nxtuc = start = buf;
    endpg = buf + strlen( buf ) -1;

 /* Locate all possible sentence terminations in this paragraph;
  * if none, print what we have as a sentence.
  */
    openbracketp=0;
    for(nxtsp = buf; *nxtsp != NULL ; nxtsp++)
      switch (*nxtsp)
	{
	case '[':
	  if ( strchr(nxtsp,']') != NULL )
	    openbracketp=nxtsp;
	  break;
	case ']':
	  if (openbracketp && n_per
	      && period[n_per-1]+4 > openbracketp
	      && strchr(".!?",*(nxtsp-1)))
	    period[n_per-1]=nxtsp-1;
	  openbracketp=0;
	  break;
	case '.':
	case '?':
	case '!':
	  if (openbracketp) continue;
	  period[n_per++] = nxtsp;
	  if (n_per >= MAXBRKS)
	    {
	      fprintf(stderr,
		      "MAXBRKS exceeded - more than %d `periods' in\n%s\n",
		      MAXBRKS, buf);
	      exit(-1);
	    }
	  break;
	default:
	  break;
	}
    
    if ( ! n_per ) {
        /* if ( endpg - buf > 3 && strchr(( endpg-2 ), ':' ) != NULL ) */
        tagSentence( buf, endpg );
	return;
    }

 /* Check each possible sentence break, using a variety of
  * heuristics...  At each stage, if evidence indicates a
  * clear decision, write the tagged sentence if appropriate,
  * and continue on to the next candidate.
  */
    for ( i=0; i<n_per; i++ )
    {
	nxtch = period[i];
	prvch = period[i] -1;

 /* For this to be a valid break, there must be a space
  * and an upper-case letter following
  */
	if (( nxtuc = strpbrk( period[i], ucs )) == NULL ||
	    ( nxtsp = strchr( period[i], ' ' )) == NULL )
	    DoNextPeriod;

 /* If a digit or other punctuation follows before the next
  * space, this cannot be a sentence break (this handles
  * medial periods in strings of initials, like "U.S.", "p.m."
  */
	while ( ++nxtch < nxtsp )
	    if ( strchr( ".,;:-?!'0123456789", *nxtch ))
		break;
	if ( nxtch < nxtsp && ( *nxtch != '\'' || strchr( " st", *(nxtch+1) )))
	    DoNextPeriod;
	else
	{
 /* Before going on, check whether nxtuc precedes nxtsp; if so,
  * this is probably a typo (space after period was elided);
  * we should fix it and continue to treat this as a candidate
  */
	    if ( nxtuc < nxtsp ) {
		for ( endwd = ++endpg; endwd > period[i]; endwd-- )
		    *(endwd+1) = *endwd;
		*(++endwd) = ' ';
		for ( j=i+1; j<n_per; j++ )
		    period[j]++;
		nxtsp = nxtuc++;
            }
        }

 /* Make sure nxtsp points as far to the right as possible
  * before checking distance to nxtuc; allowable distance is
  * up to 3 chars, to allow for intervening quote and paren.
  * (but don't allow an intervening space)
  * (Now allows for many intervening bracketed expressions as well.)
  */
	while ( *( nxtsp +1 ) == ' ' )
	    nxtsp++;
	if (( nxtuc > nxtsp + 3 ||
	      ( nxtuc == nxtsp + 3 && *( nxtuc -1 ) == ' ' ))
	    && (! (( *(nxtsp+1) == '[' )
		   && ( strchr( nxtsp, ']') +2 == nxtuc )
		   && ( strchr( ".!?", *(nxtuc-3)) == NULL))))
	    DoNextPeriod;

 /* If next token after period is a corporate abbrev, this is
  * not a break
  */
	j = k = 0;
	while ( k < MAXWDLEN && nxtuc[j] != ' ' ) {
	    if ( isalpha( nxtuc[j] ))
		nxtwd[k++] = toupper( nxtuc[j] );
	    j++;
	}
	if ( k < MAXWDLEN ) {
	    nxtwd[k] = 0;
	    for ( j=0; j<n_crp_abbrv; j++ )
		if ( !strcmp( nxtwd, crp_abbrv[j] ))
		    break;
	    if ( j < n_crp_abbrv )
		DoNextPeriod;
	} else {
	    fprintf( stderr, "TYPO? <p id=%s.%d> %s\n", idstr, pid, start );
	    DoNextPeriod;
	}

 /* Inspect the token that precedes the period
  */
	perchr = *period[i];
	*period[i] = 0;

	if (( prvsp = strrchr( start, ' ' )) != NULL )
	{

 /* This block looks at a pre-break token that is not sentence-initial.
  * Make sure we point to the first alphanumeric character, if any
  */
	    endwd = prvsp +1;
	    while ( *endwd && !isalnum( *endwd ))
		*endwd++;
	    if ( ! *endwd ) { /* This was probably an ellipsis "..." */
		*period[i] = perchr;
		tagSentence( start, nxtsp );
		start = nxtsp + 1;
		DoNextPeriod;
	    }
    
 /* - if token ends in a bracket or quote, this is a clear sentence break
  */
	    if ( strchr( "\")}]", *prvch ))
	    {
		*period[i] = perchr;
		tagSentence( start, nxtsp );
		start = nxtsp + 1;
		DoNextPeriod;
	    }

 /* - if token does not begin with upper-case, and is not a time designation
  *	("a.m" or "p.m") followed by a time-zone name, and is not "vs" or "excl",
  *	then this is a real break
  */
	    if ( !isupper( *endwd )) {
		if ( strstr( endwd, ".m" )) {
		    for ( j=0; j<n_time_zone; j++ )
			if ( !strcmp( nxtwd, time_zone[j] ))
			    break;
		    if ( j < n_time_zone ) {
			*period[i] = perchr;
			DoNextPeriod;
		    }
		}
		if ( strcmp( endwd, "vs" ) && strcmp( endwd, "excl" )) {
		    *period[i] = perchr;
		    tagSentence( start, nxtsp );
		    start = nxtsp + 1;
		}
		*period[i] = perchr;
		DoNextPeriod;
	    }

 /* - if it is one of the definite within-sentence abbrevs,
  *	this clearly is not a sentence break
  */
	    for ( j=0; j<n_mid_abbrevs; j++ )
		if ( !strcasecmp( endwd, abbrevs[j] ))
		    break;
	    if ( j < n_mid_abbrevs ) {
		*period[i] = perchr;
		DoNextPeriod;
	    }

 /* - if it is "Dr" or "St", preceded by a capitalized word,
  *	with only a space intervening, this could be a valid break,
  *	but unlikely -- just issue a warning and don't call it a break
  */
	    for ( ; j<n_abbrevs; j++ )
		if ( !strcasecmp( endwd, abbrevs[j] ))
		    break;
	    if ( j < n_abbrevs ) {
		*prvsp = 0;
		prvwd = strrchr( start, ' ' );
		if ( prvwd == NULL ) {
		    *prvsp = ' ';
		    *period[i] = perchr;
		    DoNextPeriod;
		}
		while ( *prvwd && !isalpha( *prvwd ))
		    prvwd++;
		
		if ( ! *prvwd || !isupper( *prvwd ) ||
		      strpbrk( prvwd, ",.:;\"')]}" ) ||
		      strchr( "{[(\"`", *(prvsp+1) )) {
		    *prvsp = ' ';
		    *period[i] = perchr;
		    DoNextPeriod;
		}
		*prvsp = ' ';
		*period[i] = perchr;
		fprintf( stderr, "ADR? <p id=%s.%d> %s\n", idstr, pid, start );
		DoNextPeriod;
	    }

 /* - if it is a single letter, this is almost certainly
  *	not a real break (it's a first or middle initial)
  */
	    if ( strlen( endwd ) == 1 ) {
		*period[i] = perchr;
		DoNextPeriod;
	    }

 /* At this point, we are looking at a non-initial multi-char token that
  * begins with upper-case, is not a clear mid-sentence abbrev, and is
  * followed by a capitalized word that is not a corporate abbrev.
  * If the "period" character is actually "?" or "!", OR (the token
  * contains lower case and, if a corp-abbrev, is not followed by "(")
  * then this is almost certainly a real break (if it is a corp-abbrev
  * followed by "(", this is most likely not a break)
  */
	    if ( perchr != '.' ) {
		*period[i] = perchr;
		tagSentence( start, nxtsp );
		start = nxtsp + 1;
		DoNextPeriod;
	    }
	    if ( strpbrk( endwd, lcs )) {
		for ( j=0; j<n_crp_abbrv; j++ )
		    if ( !strcasecmp( endwd, crp_abbrv[j] ))
			break;
		*period[i] = perchr;
		if ( j == n_crp_abbrv || *(nxtsp+1) != '(' ) {
		    tagSentence( start, nxtsp );
		    start = nxtsp + 1;
		}
		DoNextPeriod;
	    }

 /* Now we reach the truly ambiguous case: a sequence of upper-case
  * (possibly initials) followed by a capitalized token (e.g. "U.S.
  * Treasury" or "A.G. Edwards"; if it is an acronym (e.g. "NASA"),
  * this is most likely a real break.
  */
	    if ( strspn( endwd, ucs ) == strlen( endwd )) {
		*period[i] = perchr;
		tagSentence( start, nxtsp );
		start = nxtsp + 1;
		DoNextPeriod;
	    }

 /* Finally, we must determine whether the next token is likely to
  * be a sentence-initial word, rather than a continuation of a
  * proper name (i.e. "U.S. The" vs. "U.S. Navy" -- failing this
  * criterion does not mean we don't have a break, but the error
  * rate of calling it a non-break is diminished.  As a result, the
  * predominant error should be run-on sentences (missed breaks).
  */
	    si_node.wd = nxtwd;
	    if ( bsearch((char *)(&si_node), (char *)s_init_wd, n_s_init,
			 sizeof(si_node), w_compare ))
	    {
		*period[i] = perchr;
		tagSentence( start, nxtsp );
		start = nxtsp + 1;
	    }
	    else
	    {
		*period[i] = perchr;
		fprintf( tfp, "%s <%s.%d>\n", nxtwd, idstr, pid );
	    }
	    DoNextPeriod;

	} /* prvsp != NULL */

	else

	{ /* prvsp == NULL */
 /* This block looks at a sentence-initial token preceding
  * the period; if "period" is acually "?!", or if the token
  * looks like any kind of abbreviation, this is not a real break
  */
	    if ( perchr != '.' ) {
		*period[i] = perchr;
		tagSentence( start, nxtsp );
		start = nxtsp + 1;
		DoNextPeriod;
	    }
	    endwd = start;
	    while ( *endwd && !isalpha( *endwd ))
		endwd++;
	    if ( ! *endwd ) {
		*period[i] = perchr;
		DoNextPeriod;
	    }
	    for ( j=0; j<n_abbrevs; j++ )
		if ( !strcasecmp( endwd, abbrevs[j] ))
		    break;
	    if ( j < n_abbrevs || strlen( endwd ) == 1 || strchr( endwd, '.' )) {
		*period[i] = perchr;
		DoNextPeriod;
	    }
	    *period[i] = perchr;
	    tagSentence( start, nxtsp );
	    start = nxtsp + 1;
	    DoNextPeriod;
	}
    } /* for (i=0; i<n_per; i++ ) */

/* If there is still character data in the buffer, call it a sentence
 */
    if ( start + 2 < endpg )
      if ( *endpg == ' ')
	tagSentence( start, endpg, idstr );
      else
	tagSentence( start, (endpg + 1), idstr );
}


tagSentence( start, end )
  char *start, *end;
{
    char sent[MAXSENTLEN], *si, *so;
    int alpha, len;

    len = (end - start) + 2;
    if ( len > MAXSENTLEN )
      {
	fprintf( stderr, "Warning: in %s, ", idstr );
	fprintf(stderr,"sentence length of %d exceeds MAXSENTLEN (%d)\n",
		len,MAXSENTLEN);
	strncpy(sent,start,75);
	sent[75]=0;
	fprintf(stderr,"ignoring `sentence' beginning with:\n %s\n",
		sent);
	return;
      }

    si = start;
    so = sent;
    alpha = 0;

    while ( si < end ) {
	alpha |= (! isspace( *si ));
	*so++ = *si++;
    }
    *so = 0;

    if ( ! alpha )
	return;

    printf( "<s>\n%s\n", sent );
}

/*
unpicky_tagSentence( start, end )
  char *start, *end;
{
    if ( start >= end ) {
      fprintf( stderr, "ignoring bad sentence mark (%x !< %x) in %s\n",
	       start, end, idstr );
      fprintf( stderr, "`sentence' from start-pointer:\n%s\n", start );
      return;
    }
    printf("<s>\n");
    while ( start < end )
	putchar(*start++);
    putchar('\n');
}
*/
