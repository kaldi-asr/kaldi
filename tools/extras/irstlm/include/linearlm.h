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


// Linear discounting interpolated LMs



namespace irstlm {
	//Witten and Bell linear discounting
	class linearwb: public mdiadaptlm
	{
		int prunethresh;
	public:
		linearwb(char* ngtfile,int depth=0,int prunefreq=0,TABLETYPE tt=SHIFTBETA_B);
		int train();
		int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);
		~linearwb() {}
	};
	
	//Stupid-Backoff LM type
	class linearstb: public mdiadaptlm
	{
		int prunethresh;
	public:
		linearstb(char* ngtfile,int depth=0,int prunefreq=0,TABLETYPE tt=SHIFTBETA_B);
		int train();
		int discount(ngram ng,int size,double& fstar,double& lambda,int cv=0);
		~linearstb() {}
		int compute_backoff();
	};
	
	//Good Turing linear discounting
	//no more supported
	
}//namespace irstlm