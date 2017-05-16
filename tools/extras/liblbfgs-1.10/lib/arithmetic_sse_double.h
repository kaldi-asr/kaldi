/*
 *      SSE2 implementation of vector oprations (64bit double).
 *
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* $Id$ */

#include <stdlib.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <memory.h>

#if     1400 <= _MSC_VER
#include <intrin.h>
#endif/*1400 <= _MSC_VER*/

#if     HAVE_EMMINTRIN_H
#include <emmintrin.h>
#endif/*HAVE_EMMINTRIN_H*/

inline static void* vecalloc(size_t size)
{
#if     defined(_MSC_VER)
    void *memblock = _aligned_malloc(size, 16);
#elif   defined(__APPLE__)  /* OS X always aligns on 16-byte boundaries */
    void *memblock = malloc(size);
#else
    void *memblock = NULL, *p = NULL;
    if (posix_memalign(&p, 16, size) == 0) {
        memblock = p;
    }
#endif
    if (memblock != NULL) {
        memset(memblock, 0, size);
    }
    return memblock;
}

inline static void vecfree(void *memblock)
{
#ifdef	_MSC_VER
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

#define fsigndiff(x, y) \
    ((_mm_movemask_pd(_mm_set_pd(*(x), *(y))) + 1) & 0x002)

#define vecset(x, c, n) \
{ \
    int i; \
    __m128d XMM0 = _mm_set1_pd(c); \
    for (i = 0;i < (n);i += 8) { \
        _mm_store_pd((x)+i  , XMM0); \
        _mm_store_pd((x)+i+2, XMM0); \
        _mm_store_pd((x)+i+4, XMM0); \
        _mm_store_pd((x)+i+6, XMM0); \
    } \
}

#define veccpy(y, x, n) \
{ \
    int i; \
    for (i = 0;i < (n);i += 8) { \
        __m128d XMM0 = _mm_load_pd((x)+i  ); \
        __m128d XMM1 = _mm_load_pd((x)+i+2); \
        __m128d XMM2 = _mm_load_pd((x)+i+4); \
        __m128d XMM3 = _mm_load_pd((x)+i+6); \
        _mm_store_pd((y)+i  , XMM0); \
        _mm_store_pd((y)+i+2, XMM1); \
        _mm_store_pd((y)+i+4, XMM2); \
        _mm_store_pd((y)+i+6, XMM3); \
    } \
}

#define vecncpy(y, x, n) \
{ \
    int i; \
    for (i = 0;i < (n);i += 8) { \
        __m128d XMM0 = _mm_setzero_pd(); \
        __m128d XMM1 = _mm_setzero_pd(); \
        __m128d XMM2 = _mm_setzero_pd(); \
        __m128d XMM3 = _mm_setzero_pd(); \
        __m128d XMM4 = _mm_load_pd((x)+i  ); \
        __m128d XMM5 = _mm_load_pd((x)+i+2); \
        __m128d XMM6 = _mm_load_pd((x)+i+4); \
        __m128d XMM7 = _mm_load_pd((x)+i+6); \
        XMM0 = _mm_sub_pd(XMM0, XMM4); \
        XMM1 = _mm_sub_pd(XMM1, XMM5); \
        XMM2 = _mm_sub_pd(XMM2, XMM6); \
        XMM3 = _mm_sub_pd(XMM3, XMM7); \
        _mm_store_pd((y)+i  , XMM0); \
        _mm_store_pd((y)+i+2, XMM1); \
        _mm_store_pd((y)+i+4, XMM2); \
        _mm_store_pd((y)+i+6, XMM3); \
    } \
}

#define vecadd(y, x, c, n) \
{ \
    int i; \
    __m128d XMM7 = _mm_set1_pd(c); \
    for (i = 0;i < (n);i += 4) { \
        __m128d XMM0 = _mm_load_pd((x)+i  ); \
        __m128d XMM1 = _mm_load_pd((x)+i+2); \
        __m128d XMM2 = _mm_load_pd((y)+i  ); \
        __m128d XMM3 = _mm_load_pd((y)+i+2); \
        XMM0 = _mm_mul_pd(XMM0, XMM7); \
        XMM1 = _mm_mul_pd(XMM1, XMM7); \
        XMM2 = _mm_add_pd(XMM2, XMM0); \
        XMM3 = _mm_add_pd(XMM3, XMM1); \
        _mm_store_pd((y)+i  , XMM2); \
        _mm_store_pd((y)+i+2, XMM3); \
    } \
}

#define vecdiff(z, x, y, n) \
{ \
    int i; \
    for (i = 0;i < (n);i += 8) { \
        __m128d XMM0 = _mm_load_pd((x)+i  ); \
        __m128d XMM1 = _mm_load_pd((x)+i+2); \
        __m128d XMM2 = _mm_load_pd((x)+i+4); \
        __m128d XMM3 = _mm_load_pd((x)+i+6); \
        __m128d XMM4 = _mm_load_pd((y)+i  ); \
        __m128d XMM5 = _mm_load_pd((y)+i+2); \
        __m128d XMM6 = _mm_load_pd((y)+i+4); \
        __m128d XMM7 = _mm_load_pd((y)+i+6); \
        XMM0 = _mm_sub_pd(XMM0, XMM4); \
        XMM1 = _mm_sub_pd(XMM1, XMM5); \
        XMM2 = _mm_sub_pd(XMM2, XMM6); \
        XMM3 = _mm_sub_pd(XMM3, XMM7); \
        _mm_store_pd((z)+i  , XMM0); \
        _mm_store_pd((z)+i+2, XMM1); \
        _mm_store_pd((z)+i+4, XMM2); \
        _mm_store_pd((z)+i+6, XMM3); \
    } \
}

#define vecscale(y, c, n) \
{ \
    int i; \
    __m128d XMM7 = _mm_set1_pd(c); \
    for (i = 0;i < (n);i += 4) { \
        __m128d XMM0 = _mm_load_pd((y)+i  ); \
        __m128d XMM1 = _mm_load_pd((y)+i+2); \
        XMM0 = _mm_mul_pd(XMM0, XMM7); \
        XMM1 = _mm_mul_pd(XMM1, XMM7); \
        _mm_store_pd((y)+i  , XMM0); \
        _mm_store_pd((y)+i+2, XMM1); \
    } \
}

#define vecmul(y, x, n) \
{ \
    int i; \
    for (i = 0;i < (n);i += 8) { \
        __m128d XMM0 = _mm_load_pd((x)+i  ); \
        __m128d XMM1 = _mm_load_pd((x)+i+2); \
        __m128d XMM2 = _mm_load_pd((x)+i+4); \
        __m128d XMM3 = _mm_load_pd((x)+i+6); \
        __m128d XMM4 = _mm_load_pd((y)+i  ); \
        __m128d XMM5 = _mm_load_pd((y)+i+2); \
        __m128d XMM6 = _mm_load_pd((y)+i+4); \
        __m128d XMM7 = _mm_load_pd((y)+i+6); \
        XMM4 = _mm_mul_pd(XMM4, XMM0); \
        XMM5 = _mm_mul_pd(XMM5, XMM1); \
        XMM6 = _mm_mul_pd(XMM6, XMM2); \
        XMM7 = _mm_mul_pd(XMM7, XMM3); \
        _mm_store_pd((y)+i  , XMM4); \
        _mm_store_pd((y)+i+2, XMM5); \
        _mm_store_pd((y)+i+4, XMM6); \
        _mm_store_pd((y)+i+6, XMM7); \
    } \
}



#if     3 <= __SSE__ || defined(__SSE3__)
/*
    Horizontal add with haddps SSE3 instruction. The work register (rw)
    is unused.
 */
#define __horizontal_sum(r, rw) \
    r = _mm_hadd_ps(r, r); \
    r = _mm_hadd_ps(r, r);

#else
/*
    Horizontal add with SSE instruction. The work register (rw) is used.
 */
#define __horizontal_sum(r, rw) \
    rw = r; \
    r = _mm_shuffle_ps(r, rw, _MM_SHUFFLE(1, 0, 3, 2)); \
    r = _mm_add_ps(r, rw); \
    rw = r; \
    r = _mm_shuffle_ps(r, rw, _MM_SHUFFLE(2, 3, 0, 1)); \
    r = _mm_add_ps(r, rw);

#endif

#define vecdot(s, x, y, n) \
{ \
    int i; \
    __m128d XMM0 = _mm_setzero_pd(); \
    __m128d XMM1 = _mm_setzero_pd(); \
    __m128d XMM2, XMM3, XMM4, XMM5; \
    for (i = 0;i < (n);i += 4) { \
        XMM2 = _mm_load_pd((x)+i  ); \
        XMM3 = _mm_load_pd((x)+i+2); \
        XMM4 = _mm_load_pd((y)+i  ); \
        XMM5 = _mm_load_pd((y)+i+2); \
        XMM2 = _mm_mul_pd(XMM2, XMM4); \
        XMM3 = _mm_mul_pd(XMM3, XMM5); \
        XMM0 = _mm_add_pd(XMM0, XMM2); \
        XMM1 = _mm_add_pd(XMM1, XMM3); \
    } \
    XMM0 = _mm_add_pd(XMM0, XMM1); \
    XMM1 = _mm_shuffle_pd(XMM0, XMM0, _MM_SHUFFLE2(1, 1)); \
    XMM0 = _mm_add_pd(XMM0, XMM1); \
    _mm_store_sd((s), XMM0); \
}

#define vec2norm(s, x, n) \
{ \
    int i; \
    __m128d XMM0 = _mm_setzero_pd(); \
    __m128d XMM1 = _mm_setzero_pd(); \
    __m128d XMM2, XMM3, XMM4, XMM5; \
    for (i = 0;i < (n);i += 4) { \
        XMM2 = _mm_load_pd((x)+i  ); \
        XMM3 = _mm_load_pd((x)+i+2); \
        XMM4 = XMM2; \
        XMM5 = XMM3; \
        XMM2 = _mm_mul_pd(XMM2, XMM4); \
        XMM3 = _mm_mul_pd(XMM3, XMM5); \
        XMM0 = _mm_add_pd(XMM0, XMM2); \
        XMM1 = _mm_add_pd(XMM1, XMM3); \
    } \
    XMM0 = _mm_add_pd(XMM0, XMM1); \
    XMM1 = _mm_shuffle_pd(XMM0, XMM0, _MM_SHUFFLE2(1, 1)); \
    XMM0 = _mm_add_pd(XMM0, XMM1); \
    XMM0 = _mm_sqrt_pd(XMM0); \
    _mm_store_sd((s), XMM0); \
}


#define vec2norminv(s, x, n) \
{ \
    int i; \
    __m128d XMM0 = _mm_setzero_pd(); \
    __m128d XMM1 = _mm_setzero_pd(); \
    __m128d XMM2, XMM3, XMM4, XMM5; \
    for (i = 0;i < (n);i += 4) { \
        XMM2 = _mm_load_pd((x)+i  ); \
        XMM3 = _mm_load_pd((x)+i+2); \
        XMM4 = XMM2; \
        XMM5 = XMM3; \
        XMM2 = _mm_mul_pd(XMM2, XMM4); \
        XMM3 = _mm_mul_pd(XMM3, XMM5); \
        XMM0 = _mm_add_pd(XMM0, XMM2); \
        XMM1 = _mm_add_pd(XMM1, XMM3); \
    } \
    XMM2 = _mm_set1_pd(1.0); \
    XMM0 = _mm_add_pd(XMM0, XMM1); \
    XMM1 = _mm_shuffle_pd(XMM0, XMM0, _MM_SHUFFLE2(1, 1)); \
    XMM0 = _mm_add_pd(XMM0, XMM1); \
    XMM0 = _mm_sqrt_pd(XMM0); \
    XMM2 = _mm_div_pd(XMM2, XMM0); \
    _mm_store_sd((s), XMM2); \
}
