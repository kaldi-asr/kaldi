#include <iostream>
#include "CachedMem.cc"
using namespace std;

template <class T> 
class A : public CachedMem< A<T> > {  

public:

  T data [4];

  void info() {
    cerr << "size of class = " << sizeof(A<T>) << endl;
  }  
};

int main()
{
  A<int> * a = new A<int>;
  cerr << "A<int>: ";
  a->info();

  A<char> * b = new A<char>;
  cerr << "A<char>: ";
  b->info();

  delete a;
  delete b;

  cerr << "Stat of CachedMem<int>:" << endl;
  CachedMem< A<int> >::stat();
  cerr << "Stat of CachedMem<char>:" <<endl;
  CachedMem< A<char> >::stat();

  CachedMem< A<int> >::freeall();
  CachedMem< A<char> >::freeall();

  return 0;
}
