// $Id: verify_caching.cpp 3677 2010-10-13 09:06:51Z bertoldi $

/******************************************************************************
 IrstLM: IRST Language Model Toolkit, compile LM
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


#include <iostream>
#include <string>
#include <stdlib.h>
#include "cmd.h"
#include "util.h"
#include "mdiadapt.h"
#include "lmContainer.h"

/********************************/
using namespace std;
using namespace irstlm;

void print_help(int TypeFlag=0){
  std::cerr << std::endl << "verify_caching - verify whether caching is enabled or disabled" << std::endl;
  std::cerr << std::endl << "USAGE:"  << std::endl;
	std::cerr << "       verify_caching" << std::endl;
	std::cerr << std::endl << "DESCRIPTION:" << std::endl;
	std::cerr << std::endl << "OPTIONS:" << std::endl;
	
	FullPrintParams(TypeFlag, 0, 1, stderr);
}

void usage(const char *msg = 0)
{
  if (msg) {
    std::cerr << msg << std::endl;
  }
	if (!msg){
		print_help();
	}
}

int main(int argc, char **argv)
{	
	bool help=false;
	
  DeclareParams((char*)
								
								"Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								"h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								
								(char *)NULL
								);
	
	if (argc > 1){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	
  GetParams(&argc, &argv, (char*) NULL);
	
	if (help){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}

  if (lmContainer::is_cache_enabled()){
		std::cout << " caching is ENABLED" << std::endl;
  }else{
    std::cout << " caching is DISABLED" << std::endl;
  }

	if (mdiadaptlm::is_train_cache_enabled()){
		std::cout << " train-caching is ENABLED" << std::endl;
  }else{
    std::cout << " train-caching is DISABLED" << std::endl;
  }
}
