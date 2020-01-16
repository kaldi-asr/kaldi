
WGET ?= wget

# This commit is the latest commit available as of 2019.01.15 .
# Update it if needed.
COMMIT_ID := bc29dcfff07ced1c7a19a4ecee48e5ad583cef8e

WARP_CTC_FILENAME := ctc/warp-ctc.tar.gz
LIB_WARP_CTC := ctc/warp-ctc/build/libwarpctc.so

LDFLAGS += -Wl,-rpath=$(CURDIR)/ctc/warp-ctc/build
EXTRA_LDLIBS += $(LIB_WARP_CTC)

WITH_OMP := ON

ifdef CI_TARGETS
WITH_OMP := OFF
endif

$(LIB_WARP_CTC): $(WARP_CTC_FILENAME)
	cd ctc/warp-ctc && \
	sed -i 's/--std=c++11/-std=c++11/g' CMakeLists.txt && \
	mkdir -p build && \
	cd build && \
	cmake -DBUILD_TESTS=OFF \
	  -DWITH_OMP=$(WITH_OMP) \
		-DBUILD_SHARED=ON .. && \
	make -j2

$(WARP_CTC_FILENAME):
	cd ctc && \
	$(WGET) -O warp-ctc.tar.gz \
		--timeout=10 --tries=3 \
		https://github.com/baidu-research/warp-ctc/archive/$(COMMIT_ID).tar.gz && \
	tar xf warp-ctc.tar.gz && \
	ln -sf warp-ctc-$(COMMIT_ID) warp-ctc
