/******************************************************************************
 IrstLM: IRST Language Model Toolkit
 Copyright (C) 2006 Marcello Federico, ITC-irst Trento, Italy
 
 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.
 
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 
 ******************************************************************************/

#ifndef _WIN32_WCE
#include	<stdio.h>
#endif

#include	<stdlib.h>
#include	<ctype.h>
#include	<string.h>
#include	<stdarg.h>
#if defined(_WIN32)
#include	<windows.h>
#else
#include	<unistd.h>
#endif

#ifdef USE_UPIO
#include	"missing.h"
#include	"updef.h"
#endif

#include	"cmd.h"

#ifdef	NEEDSTRDUP
char	*strdup(const char *s);
#endif

#define	LINSIZ		10240


static Bool_T	BoolEnum[] = {
	{	(char*)"FALSE",	FALSE},
	{	(char*)"TRUE",   TRUE},
	{	(char*)"false",	FALSE},
	{	(char*)"true",   TRUE},
	{	(char*)"0",	    FALSE},
	{	(char*)"1",	     TRUE},
	{ (char*)"NO",    FALSE},
	{ (char*)"YES",    TRUE},
	{ (char*)"No",    FALSE},
	{ (char*)"Yes",    TRUE},
	{ (char*)"no",    FALSE},
	{ (char*)"yes",    TRUE},
	{ (char*)"N",     FALSE},
	{ (char*)"Y",      TRUE},
	{ (char*)"n",     FALSE},
	{ (char*)"y",      TRUE},
	END_ENUM
};


static char
*GetLine(FILE	*fp,
				 int	n,
				 char	*Line),
**str2array(char	*s,
						char	*sep);
static int
str2narray(int	type,
					 char	*s,
					 char	*sep,
					 void	**a);

static int
Scan(char	*ProgName,
		 Cmd_T	*cmds,
		 char	*Line),
SetParam(Cmd_T	*cmd,
				 char	*s),
SetEnum(Cmd_T	*cmd,
				char	*s),
SetBool(Cmd_T	*cmd,
				char	*s),
SetFlag(Cmd_T	*cmd,
				char	*s),
SetSubrange(Cmd_T	*cmd,
						char	*s),
SetStrArray(Cmd_T	*cmd,
						char	*s),
SetNumArray(Cmd_T	*cmd,
						char	*s),
SetGte(Cmd_T	*cmd,
			 char	*s),
SetLte(Cmd_T	*cmd,
			 char	*s),
CmdError(char	*opt),
EnumError(Cmd_T	*cmd,
					char	*s),
BoolError(Cmd_T	*cmd,
					char	*s),
SubrangeError(Cmd_T	*cmd,
							int	n),
GteError(Cmd_T	*cmd,
				 int	n),
LteError(Cmd_T	*cmd,
				 int	n),
PrintParam(Cmd_T	*cmd,
					 int		TypeFlag,
					 int		ValFlag,
					 FILE		*fp),
PrintParams4(int	TypeFlag,
						 int	ValFlag,
						 int	MsgFlag,
						 FILE	*fp),
FreeParam(Cmd_T	*cmd),
PrintEnum(Cmd_T	*cmd,
					int	TypeFlag,
					int	ValFlag,
					FILE	*fp),
PrintBool(Cmd_T	*cmd,
					int	TypeFlag,
					int	ValFlag,
					FILE	*fp),
PrintFlag(Cmd_T	*cmd,
					int	TypeFlag,
					int	ValFlag,
					FILE	*fp),
PrintStrArray(Cmd_T	*cmd,
							int	TypeFlag,
							int	ValFlag,
							FILE	*fp),
PrintIntArray(Cmd_T	*cmd,
							int	TypeFlag,
							int	ValFlag,
							FILE	*fp),
PrintDblArray(Cmd_T	*cmd,
							int	TypeFlag,
							int	ValFlag,
							FILE	*fp),
BuildCmdList(Cmd_T	**cmdp,
						 int	*cmdSz,
						 char	*ParName,
						 va_list	args),
StoreCmdLine(char	*s);

static Cmd_T	*pgcmds = 0;
static int	pgcmdN = 0;
static int	pgcmdSz = 0;
static char	*SepString = " \t\r\n";
static char	*ProgName = 0;
static char	**CmdLines = 0;
static int	CmdLinesSz = 0,
CmdLinesL = 0;

int 
DeclareParams(char	*ParName,
							...)
{
	va_list		args;
	
	va_start(args, ParName);
	pgcmdN = BuildCmdList(&pgcmds, &pgcmdSz, ParName, args);
	va_end(args);
	return 0;
}

int 
GetParams(int	*n,
					char	***a,
					char	*DefCmd)
{	
	char	*Line;
	int	i,
	argc = *n;
	char	**argv = *a,
	*s,
	*p,
	*defCmd;
	
#if defined(MSDOS)||defined(_WIN32)
	char	*dot = 0;
#endif
	extern char	**environ;
	
	if(!(Line=malloc(LINSIZ))) {
		fprintf(stderr, "GetParams(): Unable to alloc %d bytes\n",
						LINSIZ);
		exit(IRSTLM_CMD_ERROR_MEMORY);
	}
	for(ProgName=*argv+strlen(*argv);
			ProgName-->*argv && *ProgName!='/' && *ProgName!='\\';);
	++ProgName;
#if defined(MSDOS)||defined(_WIN32)
	if((dot=strchr(ProgName, '.'))) *dot=0;
#endif
	--argc;
	++argv;
	for(i=0; environ[i]; i++) {
		if(strncmp(environ[i], "cmd_", 4)) continue;
		strcpy(Line, environ[i]+4);
		if(!(p=strchr(Line, '='))) continue;
		*p=' ';
		StoreCmdLine(Line);
		if(Scan(ProgName, pgcmds, Line)) CmdError(environ[i]);
	}
	if((defCmd=DefCmd?(DefCmd=strdup(DefCmd)):0)) {
		defCmd += strspn(defCmd, "\n\r");
	}
	for(;;) {
		char *CmdFile = NULL;
		if(argc && argv[0][0]=='-' && argv[0][1]=='=') {
			CmdFile = argv[0]+2;
			++argv;
			--argc;
			defCmd = 0;
		}
		if(!CmdFile) {
			int	i;
			char	ch;
			
			if(!defCmd||!(i=strcspn(defCmd, "\n\r"))) break;
			ch = defCmd[i];
			defCmd[i] = 0;
			CmdFile = defCmd;
			defCmd += i+!!ch;
			defCmd += strspn(defCmd, "\n\r");
		}
		
		int	IsPipe = !strncmp(CmdFile, "@@", 2);
		
		FILE	*fp = IsPipe
		? popen(CmdFile+2, "r")
		: strcmp(CmdFile, "-")
		? fopen(CmdFile, "r")
		: stdin;
		
		
		if(!fp) {
			if(defCmd) continue;
			fprintf(stderr, "Unable to open command file %s\n", CmdFile);
			exit(IRSTLM_CMD_ERROR_IO);
		}
		while(GetLine(fp, LINSIZ, Line) && strcmp(Line, "\\End")) {
			StoreCmdLine(Line);
			if(Scan(ProgName, pgcmds, Line)) CmdError(Line);
		}
		
		if(fp!=stdin) {
			if(IsPipe)
				pclose(fp);
			else
				fclose(fp);
		}
	}
	if(DefCmd) free(DefCmd);
	
	//	while(argc && **argv=='-'){
	while(argc){
		if (**argv=='-'){
			s=strchr(*argv, '=');
			
			//allows double dash for parameters
			int dash_number=1;
			if (*(*argv+1) == '-') dash_number++; 
			if (s){	
				*s = ' ';
				if((p=strchr(*argv+dash_number, '.'))&&p<s) {
					strcpy(Line, *argv+dash_number);
				} else {
					sprintf(Line, "%s/%s", ProgName, *argv+dash_number);
				}
				*s = '=';
			}else{ //force the true value for the parameters without a value
				sprintf(Line, "%s/%s", ProgName, *argv+dash_number);
			}
			
			StoreCmdLine(Line);
			if(Scan(ProgName, pgcmds, Line)) CmdError(*argv);
			--argc;
			++argv;
		}else{ //skip tokens not starting with '-'
			--argc;
			++argv;
		}
	}
	*n = argc;
	*a = argv;
	
#if defined(MSDOS)||defined(_WIN32)
	if(dot) *dot = '.';
#endif
	free(Line);
	return 0;
}

int 
GetDotParams(char	*ParName,
						 ...)
{
	va_list		args;
	int		j,
	cmdN,
	cmdSz = 0;
	Cmd_T		*cmds = 0;
	
	va_start(args, ParName);
	cmdN = BuildCmdList(&cmds, &cmdSz, ParName, args);
	va_end(args);
	for(j=0; j<CmdLinesL; j++) Scan(ProgName, cmds, CmdLines[j]);
	for(j=0; j<cmdN; j++) FreeParam(cmds+j);
	if(cmds) free(cmds);
	return 0;
}

int 
GetStrParams(char	**lines,
						 int	n,
						 char	*parName,
						 ...)
{
	va_list		args;
	int		j,
	cmdN,
	cmdSz = 0;
	Cmd_T		*cmds = 0;
	
	va_start(args, parName);
	cmdN = BuildCmdList(&cmds, &cmdSz, parName, args);
	va_end(args);
	for(j=0; j<n; j++) Scan((char*)0, cmds, lines[j]);
	for(j=0; j<cmdN; j++) FreeParam(cmds+j);
	if(cmds) free(cmds);
	return 0;
}

int 
PrintParams(int		ValFlag,
						FILE	*fp)
{
	int TypeFlag=0;
	int MsgFlag=1;
	return PrintParams4(TypeFlag, ValFlag, MsgFlag, fp);
}

int 
FullPrintParams(int	TypeFlag,
								int	ValFlag,
								int	MsgFlag,
								FILE	*fp)
{
	return PrintParams4(TypeFlag, ValFlag, MsgFlag, fp);
}

static int 
PrintParams4(int	TypeFlag,
						 int	ValFlag,
						 int	MsgFlag,
						 FILE	*fp)
{
	int	i;
	
	fflush(fp);
	if(ValFlag) {
		fprintf(fp, "Parameters Values:\n");
	} else {
		fprintf(fp, "Parameters:\n");
	}
	for(i=0; pgcmds[i].Name; i++) {
		PrintParam(pgcmds+i, TypeFlag, ValFlag, fp);
		if(MsgFlag&&pgcmds[i].Msg) {
			char	*s=pgcmds[i].Msg,
			*p;
			for(;(p=strchr(s, '\n')); s=++p) {
				fprintf(fp, "%6s%*.*s\n", "", (int)(p-s), (int)(p-s), s);
			}
			if(s) fprintf(fp, "%6s%s", "", s);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	fflush(fp);
	return 0;
}

int 
SPrintParams(char	***a,
						 char	*pfx)
{
	int	l,
	n;
	Cmd_T	*cmd;
	
	if(!pfx) pfx="";
	l = strlen(pfx);
	for(n=0, cmd=pgcmds; cmd->Name; cmd++) n += !!cmd->ArgStr;
	a[0] = calloc(n, sizeof(char*));
	for(n=0, cmd=pgcmds; cmd->Name; cmd++) {
		if(!cmd->ArgStr) continue;
		a[0][n] = malloc(strlen(cmd->Name)+strlen(cmd->ArgStr)+l+2);
		sprintf(a[0][n], "%s%s=%s", pfx, cmd->Name, cmd->ArgStr);
		++n;
	}
	return n;
}

static int 
BuildCmdList(Cmd_T	**cmdp,
						 int	*cmdSz,
						 char	*ParName,
						 va_list	args)
{
	int	j,
	c,
	cmdN=0;
	char	*s;
	Cmd_T	*cmd,
	*cmds;
	
	if(!*cmdSz) {
		if(!(cmds=*cmdp=malloc((1+(*cmdSz=BUFSIZ))*sizeof(Cmd_T)))) {
			fprintf(stderr, "BuildCmdList(): malloc() failed\n");
			exit(IRSTLM_CMD_ERROR_MEMORY);
		}
	} else {
		for(cmds=*cmdp; cmds[cmdN].Name; ++cmdN);
	}
	while(ParName) {
		if(cmdN==*cmdSz) {
			cmds=*cmdp=realloc(cmds,
												 (1+(*cmdSz+=BUFSIZ))*sizeof(Cmd_T));
			if(!cmds) {
				fprintf(stderr,
								"BuildCmdList(): realloc() failed\n");
				exit(IRSTLM_CMD_ERROR_MEMORY);
			}
		}
		for(j=0; j<cmdN&&strcmp(cmds[j].Name, ParName)<0; j++);
		for(c=cmdN; c>j; c--) cmds[c] = cmds[c-1];
		cmd = cmds+j;
		cmd->Name = ParName;
		cmd->Type = va_arg(args, int);
		cmd->Val = va_arg(args, void*);
		cmd->Msg = 0;
		cmd->Flag = 0;
		cmd->p = 0;
		
		switch(cmd->Type&~CMDMSG) {
			case CMDENUMTYPE:	/* get the pointer to Enum_T struct */
			case CMDFLAGTYPE:
				cmd->p = va_arg(args, void*);
				break;
			case CMDSUBRANGETYPE:	/* get the two limits		 */
				cmd->p = (void*)calloc(2, sizeof(int));
				((int*)cmd->p)[0] = va_arg(args, int);
				((int*)cmd->p)[1] = va_arg(args, int);
				break;
			case CMDGTETYPE:	/* lower or upper bound	 	*/
			case CMDLTETYPE:
				cmd->p = (void*)calloc(1, sizeof(int));
				((int*)cmd->p)[0] = va_arg(args, int);
				break;
			case CMDSTRARRAYTYPE:	/* separator string	 	*/
				cmd->p = (s=va_arg(args, char*)) ? (void*)strdup(s) : 0;
				break;
			case CMDDBLARRAYTYPE:
			case CMDINTARRAYTYPE: /* separator & pointer to length	*/
				cmd->p = (void*)calloc(2, sizeof(void*));
				s = va_arg(args, char*);
				((char**)cmd->p)[0] = s ? strdup(s) : 0;
				((int**)cmd->p)[1] = va_arg(args, int*);
				*((int**)cmd->p)[1] = 0;
				break;
			case CMDBOOLTYPE:
				cmd->p = BoolEnum;
				break;
				//cmd->p = (Bool_T*)calloc(1, sizeof(Bool_T));
				//				cmd->p = va_arg(args, void*);
//				cmd->p = BoolEnum;
			case CMDDOUBLETYPE:	/* nothing else is needed	 */
			case CMDFLOATTYPE:
			case CMDINTTYPE:
			case CMDSTRINGTYPE:
				break;
			default:
				fprintf(stderr, "%s: %s %d %s \"%s\"\n",
								"BuildCmdList()", "Unknown Type",
								cmd->Type&~CMDMSG, "for parameter", cmd->Name);
				exit(IRSTLM_CMD_ERROR_DATA);
		}
		if(cmd->Type&CMDMSG) {
			cmd->Type&=~CMDMSG;
			cmd->Msg = va_arg(args, char*);
		}
		cmdN++;
		ParName = va_arg(args, char*);
	}
	cmds[cmdN].Name = 0;
	return cmdN;
}

static int 
CmdError(char	*opt)
{
	fprintf(stderr, "Invalid option \"%s\"\n", opt);
	fprintf(stderr, "This program expects the following parameters:\n");
	PrintParams4(TRUE, FALSE, TRUE, stderr);
	exit(IRSTLM_CMD_ERROR_DATA);
	return 0;
}

static int 
FreeParam(Cmd_T	*cmd)
{
	switch(cmd->Type) {
		case CMDBOOLTYPE2:
		case CMDSUBRANGETYPE:
		case CMDGTETYPE:
		case CMDLTETYPE:
		case CMDSTRARRAYTYPE:
			if(cmd->p) free(cmd->p);
			break;
		case CMDINTARRAYTYPE:
		case CMDDBLARRAYTYPE:
			if(!cmd->p) break;
			if(*(char**)cmd->p) free(*(char**)cmd->p);
			free(cmd->p);
			break;
	}
	return 0;
}

static int 
PrintParam(Cmd_T	*cmd,
					 int		TypeFlag,
					 int		ValFlag,
					 FILE		*fp)
{
	char	ts[128];
	
	*ts=0;
	fprintf(fp, "%4s", "");
	switch(cmd->Type) {
		case CMDDOUBLETYPE:
			fprintf(fp, "%s", cmd->Name);
			if(TypeFlag) fprintf(fp, " [double]");
			if(ValFlag) fprintf(fp, ": %22.15e", *(double*)cmd->Val);
			break;
		case CMDFLOATTYPE:
			fprintf(fp, "%s", cmd->Name);
			if(TypeFlag) fprintf(fp, " [float]");
			if(ValFlag) fprintf(fp, ": %22.15e", *(float *)cmd->Val);
			break;
		case CMDBOOLTYPE2:
		case CMDBOOLTYPE:
			PrintBool(cmd, TypeFlag, ValFlag, fp);
			break;
		case CMDENUMTYPE:
			PrintEnum(cmd, TypeFlag, ValFlag, fp);
			break;
		case CMDFLAGTYPE:
			PrintFlag(cmd, TypeFlag, ValFlag, fp);
			break;
		case CMDINTTYPE:
			if(TypeFlag) sprintf(ts, " [int]");
		case CMDSUBRANGETYPE:
			if(TypeFlag&&!*ts) sprintf(ts, " [int %d ... %d]",
																 ((int*)cmd->p)[0],
																 ((int*)cmd->p)[1]);
		case CMDGTETYPE:
			if(TypeFlag&&!*ts) sprintf(ts, " [int >= %d]",
																 ((int*)cmd->p)[0]);
		case CMDLTETYPE:
			if(TypeFlag&&!*ts) sprintf(ts, " [int <= %d]",
																 ((int*)cmd->p)[0]);
			fprintf(fp, "%s", cmd->Name);
			if(*ts) fprintf(fp, " %s", ts);
			if(ValFlag) fprintf(fp, ": %d", *(int*)cmd->Val);
			break;
		case CMDSTRINGTYPE:
			fprintf(fp, "%s", cmd->Name);
			if(TypeFlag) fprintf(fp, " [string]");
			if(ValFlag) {
				if(*(char **)cmd->Val) {
					fprintf(fp, ": \"%s\"", *(char**)cmd->Val);
				} else {
					fprintf(fp, ": %s", "NULL");
				}
			}
			break;
		case CMDSTRARRAYTYPE:
			PrintStrArray(cmd, TypeFlag, ValFlag, fp);
			break;
		case CMDINTARRAYTYPE:
			PrintIntArray(cmd, TypeFlag, ValFlag, fp);
			break;
		case CMDDBLARRAYTYPE:
			PrintDblArray(cmd, TypeFlag, ValFlag, fp);
			break;
		default:
			fprintf(stderr, "%s: %s %d %s \"%s\"\n",
							"PrintParam",
							"Unknown Type",
							cmd->Type,
							"for parameter",
							cmd->Name);
			exit(IRSTLM_CMD_ERROR_DATA);
	}
	fprintf(fp, ":");
	//	fprintf(fp, "\n");
	fflush(fp);
	return 0;
}

static char *
GetLine(FILE	*fp,
				int	n,
				char	*Line)
{
	int	j,
	l,
	offs=0;
	
	for(;;) {
		if(!fgets(Line+offs, n-offs, fp)) {
			return 0;
		}
		if(Line[offs]=='#') continue;
		l = strlen(Line+offs)-1;
		Line[offs+l] = 0;
		for(j=offs; Line[j]&&isspace((unsigned char)Line[j]); j++,l--);
		if(l<1) continue;
		if(j > offs) {
			char	*s = Line+offs,
			*q = Line+j;
			
			while((*s++=*q++))
				;
		}
		if(Line[offs+l-1]=='\\') {
			offs += l;
			Line[offs-1] = ' ';
		} else {
			break;
		}
	}
	return Line;
}

static int 
Scan(char	*ProgName,
		 Cmd_T	*cmds,
		 char	*Line)
{
	char	*q,
	*p;
	int	i,
	hl,
	HasToMatch = FALSE,
	c0,
	c;
	
	p = Line+strspn(Line, SepString);
	if(!(hl=strcspn(p, SepString))) return 0;
	if(ProgName&&(q=strchr(p, '/')) && q-p<hl) {
		*q = 0;
		if(strcmp(p, ProgName)) {
			*q = '/';
			return 0;
		}
		*q = '/';
		HasToMatch=TRUE;
		p = q+1;
	}
	if(!(hl=strcspn(p, SepString))) return 0;
	c0 = p[hl];
	p[hl] = 0;
	for(i=0, c=1; cmds[i].Name&&(c=strcmp(cmds[i].Name, p))<0; i++)
		;
	p[hl] = c0;
	if(!c) return SetParam(cmds+i, p+hl+strspn(p+hl, SepString));
	return HasToMatch && c;
}

static int 
SetParam(Cmd_T	*cmd,
				 char	*_s)
{
	char *s;
	
	if(!*_s && cmd->Type==CMDENUMTYPE && cmd->Flag==1){
		s=(char*) malloc(5);
		strcpy(s,"TRUE");
	}else{
		s=_s;
	}
	
	if (!*s || (s=='\0' && cmd->Flag==0)){
		fprintf(stderr,
						"WARNING: No value specified for parameter \"%s\"\n",
						cmd->Name);
		return 0;
	}
	
	switch(cmd->Type) {
		case CMDDOUBLETYPE:
			if(sscanf(s, "%lf", (double*)cmd->Val)!=1) {
				fprintf(stderr,
								"Float value required for parameter \"%s\"\n",
								cmd->Name);
				exit(IRSTLM_CMD_ERROR_DATA);
			}
			break;
		case CMDFLOATTYPE:
			if(sscanf(s, "%f", (float*)cmd->Val)!=1) {
				fprintf(stderr,
								"Float value required for parameter \"%s\"\n",
								cmd->Name);
				exit(IRSTLM_CMD_ERROR_DATA);
			}
			break;
		case CMDBOOLTYPE2:
		case CMDBOOLTYPE:
			SetBool(cmd, s);
			break;
		case CMDENUMTYPE:
			SetEnum(cmd, s);
			break;
		case CMDFLAGTYPE:
			SetFlag(cmd, s);
			break;
		case CMDINTTYPE:
			/*They are the same when used for output, e.g. with printf, but different when used as input specifier e.g. with scanf, where %d scans an integer as a signed decimal number, but %i defaults to decimal but also allows hexadecimal (if preceded by "0x") and octal if preceded by "0".
			 So "033" would be 27 with %i but 33 with %d.
			 */
			if(sscanf(s, "%d", (int*)cmd->Val)!=1) {
				fprintf(stderr,
								"Integer value required for parameter \"%s\"\n",
								cmd->Name);
				exit(IRSTLM_CMD_ERROR_DATA);
			}
			break;
		case CMDSTRINGTYPE:
			*(char **)cmd->Val = (strcmp(s, "<NULL>") && strcmp(s, "NULL"))
			? strdup(s)
			: 0;
			break;
		case CMDSTRARRAYTYPE:
			SetStrArray(cmd, s);
			break;
		case CMDINTARRAYTYPE:
		case CMDDBLARRAYTYPE:
			SetNumArray(cmd, s);
			break;
		case CMDGTETYPE:
			SetGte(cmd, s);
			break;
		case CMDLTETYPE:
			SetLte(cmd, s);
			break;
		case CMDSUBRANGETYPE:
			SetSubrange(cmd, s);
			break;
		default:
			fprintf(stderr, "%s: %s %d %s \"%s\"\n",
							"SetParam",
							"Unknown Type",
							cmd->Type,
							"for parameter",
							cmd->Name);
			exit(IRSTLM_CMD_ERROR_DATA);
	}
	cmd->ArgStr = strdup(s);
	
	if(!*_s && cmd->Type==CMDENUMTYPE && cmd->Flag==1){
		free (s);
	}
	
	return 0;
}

static int 
SetBool(Cmd_T	*cmd,
				char	*s)
{
	Bool_T	*en;
	
	for(en=(Bool_T*)cmd->p; en->Name; en++) {
		if(*en->Name && !strcmp(s, en->Name)) {
			*(char*)cmd->Val = en->Idx;
			return 0;
		}
	}
	return BoolError(cmd, s);
}


static int 
SetEnum(Cmd_T	*cmd,
				char	*s)
{
	Enum_T	*en;
	
	for(en=(Enum_T*)cmd->p; en->Name; en++) {
		if(*en->Name && !strcmp(s, en->Name)) {
			*(int*)cmd->Val = en->Idx;
			return 0;
		}
	}
	return EnumError(cmd, s);
}

int
EnumIdx(Enum_T	*en,
				char	*s)
{
	if(en) for(; en->Name; en++) {
		if(*en->Name && !strcmp(s, en->Name)) return en->Idx;
	}
	return -1;
}

char
BoolIdx(Bool_T	*en,
				char	*s)
{
	if(en) for(; en->Name; en++) {
		if(*en->Name && !strcmp(s, en->Name)) return en->Idx;
	}
	return -1;
}

char *
EnumStr(Enum_T	*en,
				int	i)
{
	if(en) for(; en->Name; en++) if(en->Idx==i) return en->Name;
	return 0;
}

char *
BoolStr(Bool_T	*en,
				int	i)
{
	if(en) for(; en->Name; en++) if(en->Idx==i) return en->Name;
	return 0;
}

static int 
SetFlag(Cmd_T	*cmd,
				char	*s)
{
	Enum_T	*en;
	int	l;
	
	for(; (l=strcspn(s, "+"))>0; s+=l,s+=!!*s) {
		for(en=(Enum_T*)cmd->p;
				en->Name&&(l!=strlen(en->Name)||strncmp(s, en->Name, l));
				en++);
		if(!en->Name) return EnumError(cmd, s);
		*(int*)cmd->Val |= en->Idx;
	}
	return 0;
}

static int 
SetSubrange(Cmd_T	*cmd,
						char	*s)
{
	int	n;

	/*They are the same when used for output, e.g. with printf, but different when used as input specifier e.g. with scanf, where %d scans an integer as a signed decimal number, but %i defaults to decimal but also allows hexadecimal (if preceded by "0x") and octal if preceded by "0".
	 So "033" would be 27 with %i but 33 with %d.
	 */
	if(sscanf(s, "%d", &n)!=1) {
		fprintf(stderr,
						"Integer value required for parameter \"%s\"\n",
						cmd->Name);
		exit(IRSTLM_CMD_ERROR_DATA);
	}
	if(n < *(int*)cmd->p || n > *((int*)cmd->p+1)) {
		return SubrangeError(cmd, n);
	}
	*(int*)cmd->Val = n;
	return 0;
}

static int 
SetGte(Cmd_T	*cmd,
			 char	*s)
{
	int	n;
	
	/*They are the same when used for output, e.g. with printf, but different when used as input specifier e.g. with scanf, where %d scans an integer as a signed decimal number, but %i defaults to decimal but also allows hexadecimal (if preceded by "0x") and octal if preceded by "0".
		So "033" would be 27 with %i but 33 with %d.
		*/
	if(sscanf(s, "%d", &n)!=1) {
		fprintf(stderr,
						"Integer value required for parameter \"%s\"\n",
						cmd->Name);
		exit(IRSTLM_CMD_ERROR_DATA);
	}
	if(n<*(int*)cmd->p) {
		return GteError(cmd, n);
	}
	*(int*)cmd->Val = n;
	return 0;
}

static int 
SetStrArray(Cmd_T	*cmd,
						char	*s)
{
	*(char***)cmd->Val = str2array(s, (char*)cmd->p);
	return 0;
}

static int 
SetNumArray(Cmd_T	*cmd,
						char	*s)
{
	*((int**)cmd->p)[1] = str2narray(cmd->Type, s,
																	 *((char**)cmd->p), cmd->Val);
	return 0;
}

static int 
SetLte(Cmd_T	*cmd,
			 char	*s)
{
	int	n;
	
	/*They are the same when used for output, e.g. with printf, but different when used as input specifier e.g. with scanf, where %d scans an integer as a signed decimal number, but %i defaults to decimal but also allows hexadecimal (if preceded by "0x") and octal if preceded by "0".
	 So "033" would be 27 with %i but 33 with %d.
	 */
	if(sscanf(s, "%d", &n)!=1) {
		fprintf(stderr,
						"Integer value required for parameter \"%s\"\n",
						cmd->Name);
		exit(IRSTLM_CMD_ERROR_DATA);
	}
	if(n > *(int*)cmd->p) {
		return LteError(cmd, n);
	}
	*(int*)cmd->Val = n;
	return 0;
}

static int 
EnumError(Cmd_T	*cmd,
					char	*s)
{
	Enum_T	*en;
	
	fprintf(stderr,
					"Invalid value \"%s\" for parameter \"%s\"\n", s, cmd->Name);
	fprintf(stderr, "Valid values are:\n");
	for(en=(Enum_T*)cmd->p; en->Name; en++) {
		if(*en->Name) fprintf(stderr, " %s\n", en->Name);
	}
	fprintf(stderr, "\n");
	exit(IRSTLM_CMD_ERROR_DATA);
	return 0;
}

static int 
BoolError(Cmd_T	*cmd,
					char	*s)
{
	Bool_T	*en;
	
	fprintf(stderr,
					"Invalid value \"%s\" for parameter \"%s\"\n", s, cmd->Name);
	fprintf(stderr, "Valid values are:\n");
	for(en=(Bool_T*)cmd->p; en->Name; en++) {
		if(*en->Name) fprintf(stderr, " %s\n", en->Name);
	}
	fprintf(stderr, "\n");
	exit(IRSTLM_CMD_ERROR_DATA);
	return 0;
}

static int 
GteError(Cmd_T	*cmd,
				 int	n)
{
	fprintf(stderr,
					"Value %d out of range for parameter \"%s\"\n", n, cmd->Name);
	fprintf(stderr, "Valid values must be greater than or equal to %d\n",
					*(int*)cmd->p);
	exit(IRSTLM_CMD_ERROR_DATA);
	return 0;
}

static int 
LteError(Cmd_T	*cmd,
				 int	n)
{
	fprintf(stderr,
					"Value %d out of range for parameter \"%s\"\n", n, cmd->Name);
	fprintf(stderr, "Valid values must be less than or equal to %d\n",
					*(int*)cmd->p);
	exit(IRSTLM_CMD_ERROR_DATA);
	return 0;
}

static int 
SubrangeError(Cmd_T	*cmd,
							int	n)
{
	fprintf(stderr,
					"Value %d out of range for parameter \"%s\"\n", n, cmd->Name);
	fprintf(stderr, "Valid values range from %d to %d\n",
					*(int*)cmd->p, *((int*)cmd->p+1));
	exit(IRSTLM_CMD_ERROR_DATA);
	return 0;
}

static int 
PrintEnum(Cmd_T	*cmd,
					int	TypeFlag,
					int	ValFlag,
					FILE	*fp)
{
	Enum_T	*en;
	
	fprintf(fp, "%s", cmd->Name);
	if(TypeFlag) {
		fprintf(fp, " [enum { ");
		
		char	*sep="";
		
		for(en=(Enum_T*)cmd->p; en->Name; en++) {
			if(*en->Name) {
				fprintf(fp, "%s%s", sep, en->Name);
				sep=", ";
			}
		}
		fprintf(fp, " }]");
	}
	if(ValFlag) {
		for(en=(Enum_T*)cmd->p; en->Name; en++) {
			if(*en->Name && en->Idx==*(int*)cmd->Val) {
				fprintf(fp, ": %s", en->Name);
			}
		}
	}
	//	fprintf(fp, "\n");
	return 0;
}

static int 
PrintBool(Cmd_T	*cmd,
					int	TypeFlag,
					int	ValFlag,
					FILE	*fp)
{
	Bool_T	*en;
	
	fprintf(fp, "%s", cmd->Name);
	if(TypeFlag) {
		fprintf(fp, " [enum { ");
		
		char	*sep="";
		
		for(en=(Bool_T*)cmd->p; en->Name; en++) {
			if(*en->Name) {
				fprintf(fp, "%s%s", sep, en->Name);
				sep=", ";
			}
		}
		fprintf(fp, " }]");
	}
	if(ValFlag) {
		for(en=(Bool_T*)cmd->p; en->Name; en++) {
			if(*en->Name && en->Idx==*(int*)cmd->Val) {
				fprintf(fp, ": %s", en->Name);
			}
		}
	}
	//	fprintf(fp, "\n");
	return 0;
}

static int 
PrintFlag(Cmd_T	*cmd,
					int	TypeFlag,
					int	ValFlag,
					FILE	*fp)
{
	Enum_T	*en;
	char	*sep="";
	
	fprintf(fp, "%s", cmd->Name);
	if(TypeFlag) {
		fprintf(fp, ": flag { ");
		for(en=(Enum_T*)cmd->p; en->Name; en++) {
			if(*en->Name) {
				fprintf(fp, "%s%s", sep, en->Name);
				sep=", ";
			}
		}
		fprintf(fp, " }");
	}
	if(ValFlag) {
		fprintf(fp, ": ");
		for(en=(Enum_T*)cmd->p; en->Name; en++) {
			if(*en->Name && (en->Idx&*(int*)cmd->Val)==en->Idx) {
				fprintf(fp, "%s%s", sep, en->Name);
				sep="+";
			}
		}
	}
	fprintf(fp, "\n");
	return 0;
}

static int 
PrintStrArray(Cmd_T	*cmd,
							int	TypeFlag,
							int	ValFlag,
							FILE	*fp)
{
	char	*indent,
	**s = *(char***)cmd->Val;
	int	l = 4+strlen(cmd->Name);
	
	fprintf(fp, "%s", cmd->Name);
	if(TypeFlag) {
		fprintf(fp, ": string array, separator \"%s\"",
						cmd->p?(char*)cmd->p:"");
	}
	indent = malloc(l+2);
	memset(indent, ' ', l+1);
	indent[l+1] = 0;
	if(ValFlag) {
		fprintf(fp, ": %s", s ? (*s ? *s++ : "NULL") : "");
		if(s) while(*s) {
			fprintf(fp, "\n%s %s", indent, *s++);
		}
	}
	free(indent);
	fprintf(fp, "\n");
	return 0;
}

static int 
PrintIntArray(Cmd_T	*cmd,
							int	TypeFlag,
							int	ValFlag,
							FILE	*fp)
{
	char	*indent;
	int	l = 4+strlen(cmd->Name),
	n,
	*i = *(int**)cmd->Val;
	
	fprintf(fp, "%s", cmd->Name);
	if(TypeFlag) {
		fprintf(fp, ": int array, separator \"%s\"",
						*(char**)cmd->p?*(char**)cmd->p:"");
	}
	n = *((int**)cmd->p)[1];
	indent = malloc(l+2);
	memset(indent, ' ', l+1);
	indent[l+1] = 0;
	if(ValFlag) {
		fprintf(fp, ":");
		if(i&&n>0) {
			fprintf(fp, " %d", *i++);
			while(--n) fprintf(fp, "\n%s %d", indent, *i++);
		}
	}
	free(indent);
	fprintf(fp, "\n");
	return 0;
}

static int 
PrintDblArray(Cmd_T	*cmd,
							int	TypeFlag,
							int	ValFlag,
							FILE	*fp)
{
	char	*indent;
	int	l = 4+strlen(cmd->Name),
	n;
	double	*x = *(double**)cmd->Val;
	
	fprintf(fp, "%s", cmd->Name);
	if(TypeFlag) {
		fprintf(fp, ": double array, separator \"%s\"",
						*(char**)cmd->p?*(char**)cmd->p:"");
	}
	n = *((int**)cmd->p)[1];
	indent = malloc(l+2);
	memset(indent, ' ', l+1);
	indent[l+1] = 0;
	if(ValFlag) {
		fprintf(fp, ":");
		if(x&&n>0) {
			fprintf(fp, " %e", *x++);
			while(--n) fprintf(fp, "\n%s %e", indent, *x++);
		}
	}
	free(indent);
	fprintf(fp, "\n");
	return 0;
}

static char **
str2array(char	*s,
					char	*sep)
{
	char	*p, **a;
	int	n = 0;
	
	if(!sep) sep = SepString;
	p = s += strspn(s, sep);
	if(!*p) return 0;
	while(*p) {
		p += strcspn(p, sep);
		p += strspn(p, sep);
		++n;
	}
	a = calloc(n+1, sizeof(char*));
	p = s;
	n = 0;
	while(*p) {
		int l = strcspn(p, sep);
		a[n] = malloc(l+1);
		memcpy(a[n], p, l);
		a[n][l] = 0;
		++n;
		p += l;
		p += strspn(p, sep);
	}
	return a;
}

int
str2narray(int	type,
					 char	*s,
					 char	*sep,
					 void	**a)
{
	char	*p;
	double	*x;
	int	*i;
	int	n = 0;
	
	if(!sep) sep=SepString;
	for(p=s; *p; ) {
		p += strcspn(p, sep);
		p += !!*p;
		++n;
	}
	*a = 0;
	if(!n) return 0;
	*a = calloc(n, (type==CMDINTARRAYTYPE)?sizeof(int):sizeof(double));
	i = (int*)*a;
	x = (double*)*a;
	p = s;
	n = 0;
	while(*p) {
		switch(type) {
			case CMDINTARRAYTYPE:
				*i++ = atoi(p);
				break;
			case CMDDBLARRAYTYPE:
				*x++ = atof(p);
				break;
		}
		++n;
		p += strcspn(p, sep);
		p += !!*p;
	}
	return n;
}

static int 
StoreCmdLine(char	*s)
{
	s += strspn(s, SepString);
	if(!*s) return 0;
	if(CmdLinesL>=CmdLinesSz) {
		CmdLines=CmdLinesSz
		? (char**)realloc(CmdLines,
											(CmdLinesSz+=BUFSIZ)*sizeof(char**))
		: (char**)malloc((CmdLinesSz=BUFSIZ)*sizeof(char**));
		if(!CmdLines) {
			fprintf(stderr, "%s\n",
							"StoreCmdLine(): malloc() failed");
			exit(IRSTLM_CMD_ERROR_MEMORY);
		}
	}
	CmdLines[CmdLinesL++] = strdup(s);
	return 0;
}

