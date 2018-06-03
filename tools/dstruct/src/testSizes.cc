
#include <stdio.h>
#include <sys/types.h>
#include <Trie.cc>
#include <Counts.h>
#include <XCount.h>
#include <Vocab.h>
#include <NgramStats.h>


#if defined(__SUNPRO_CC)
# pragma pack(2)
#endif
class foo {
public:
#if defined(__INTEL_COMPILER) || defined(__GNUC__)
	int x __attribute__ ((packed));
#else
	int x;
#endif
	short y;
};
#if defined(__SUNPRO_CC)
# pragma pack()
#endif

class bar {
public:
	foo x;
	short y;
};



int main() 
{
	bar b;

	printf("sizeof(void *) = %lu, sizeof(long) = %lu sizeof(size_t) = %lu\n",
		(unsigned long)sizeof(void *), (unsigned long)sizeof(long),
		(unsigned long)sizeof(size_t));
	printf("sizeof class foo = %lu, bar = %lu\n",
		(unsigned long)sizeof(foo), (unsigned long)sizeof(bar));

	printf("sizeof Trie<short,short> = %lu\n", (unsigned long)sizeof(Trie<short,short>));
	printf("sizeof Trie<int,int> = %lu\n", (unsigned long)sizeof(Trie<int,int>));
	printf("sizeof Trie<long,long> = %lu\n", (unsigned long)sizeof(Trie<long,long>));
	printf("sizeof Trie<unsigned,Count> = %lu\n", (unsigned long)sizeof(Trie<unsigned,Count>));
	printf("sizeof Trie<unsigned,XCount> = %lu\n", (unsigned long)sizeof(Trie<unsigned,XCount>));
	printf("sizeof Trie<short,double> = %lu\n", (unsigned long)sizeof(Trie<short,double>));
	printf("sizeof VocabIndex = %lu\n", (unsigned long)sizeof(VocabIndex));
	printf("sizeof NgramCount = %lu\n", (unsigned long)sizeof(NgramCount));
	printf("sizeof Trie<VocabIndex,NgramCount> = %lu\n", (unsigned long)sizeof(Trie<VocabIndex,NgramCount>));

	//b.x.x = 1;
	//b.y = 2;
}
