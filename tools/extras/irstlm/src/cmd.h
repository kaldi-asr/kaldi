// $Id: cmd.h 3626 2010-10-07 11:41:05Z bertoldi $

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

#if !defined(CMD_H)

#define	CMD_H



#define	FALSE	0
#define	TRUE	1

#define END_ENUM    {   (char*)0,  0 }



#define IRSTLM_CMD_NO_ERROR         0
#define IRSTLM_CMD_ERROR_GENERIC    1
#define IRSTLM_CMD_ERROR_IO         2
#define IRSTLM_CMD_ERROR_MEMORY     3
#define IRSTLM_CMD_ERROR_DATA       4
#define IRSTLM_CMD_ERROR_MODEL      5


#define	CMDDOUBLETYPE	1
#define	CMDENUMTYPE	2
#define	CMDINTTYPE	3
#define	CMDSTRINGTYPE	4
#define	CMDSUBRANGETYPE	5
#define	CMDGTETYPE	6
#define	CMDLTETYPE	7
#define	CMDSTRARRAYTYPE	8
#define	CMDBOOLTYPE	9
#define	CMDBOOLTYPE2	19
#define	CMDFLAGTYPE	10
#define	CMDINTARRAYTYPE	11
#define	CMDDBLARRAYTYPE	12
#define	CMDFLOATTYPE	13

#define CMDMSG		(1<<31)

#include <stdio.h>

#ifdef	__cplusplus
extern "C" {
#endif
	
	typedef struct {
		char *Name;
		int	Idx;
	} Enum_T;
	
	typedef struct {
		char *Name;
		char Idx;
	} Bool_T;
	
	typedef struct {
		int Type;
		int Flag;
		void *Val;
		void *p;
		char *Name;
		char *ArgStr;
		char *Msg;
	} Cmd_T;
	
	int
	DeclareParams(char	*,
								...),
	GetParams(int	*n,
						char	***a,
						char	*CmdFileName),
	GetDotParams(char	*,
							 ...),
	SPrintParams(char	***a,
							 char	*pfx),
	PrintParams(int		ValFlag,
							FILE	*fp),
	FullPrintParams(int		TypeFlag,
							int		ValFlag,
							int		MsgFlag,
									FILE	*fp),
	EnumIdx(Enum_T	*en,
					char	*s);
	char BoolIdx(Bool_T	*en,
					char	*s);
	char
	*EnumStr(Enum_T	*en,
					 int	i);
	char
	*BoolStr(Bool_T	*en,
					 int	i);
	
#ifdef	__cplusplus
}
#endif

#endif

