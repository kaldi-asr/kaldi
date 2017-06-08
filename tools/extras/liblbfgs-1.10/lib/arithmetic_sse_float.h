/*
 *      SSE/SSE3 implementation of vector oprations (32bit float).
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
#endif/*_MSC_VER*/

#if     HAVE_XMMINTRIN_H
#include <xmmintrin.h>
#endif/*HAVE_XMMINTRIN_H*/

#if     LBFGS_FLOAT == 32 && LBFGS_IEEE_FLOAT
#define fsigndiff(x, y) (((*(uint32_t*)(x)) ^ (*(uint32_t*)(y))) & 0x80000000U)
#else
#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)
#endif/*LBFGS_IEEE_FLOAT*/

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
    _aligned_free(memblock);
}

#define vecset(x, c, n) \
{ \
    int i; \
    __m128 XMM0 = _mm_set_ps1(c); \
    for (i = 0;i < (n);i += 16) { \
        _mm_store_ps((x)+i   , XMM0); \
        _mm_store_ps((x)+i+ 4, XMM0); \
        _mm_store_ps((x)+i+ 8, XMM0); \
        _mm_store_ps((x)+i+12, XMM0); \
    } \
}

#define veccpy(y, x, n) \
{ \
    int i; \
    for (i = 0;i < (n);i += 16) { \
        __m128 XMM0 = _mm_load_ps((x)+i   ); \
        __m128 XMM1 = _mm_load_ps((x)+i+ 4); \
        __m128 XMM2 = _mm_load_ps((x)+i+ 8); \
        __m128 XMM3 = _mm_load_ps((x)+i+12); \
        _mm_store_ps((y)+i   , XMM0); \
        _mm_store_ps((y)+i+ 4, XMM1); \
        _mm_store_ps((y)+i+ 8, XMM2); \
        _mm_store_ps((y)+i+12, XMM3); \
    } \
}

#define vecncpy(y, x, n) \
{ \
    int i; \
    const uint32_t mask = 0x80000000; \
    __m128 XMM4 = _mm_load_ps1((float*)&mask); \
    for (i = 0;i < (n);i += 16) { \
        __m128 XMM0 = _mm_load_ps((x)+i   ); \
        __m128 XMM1 = _mm_load_ps((x)+i+ 4); \
        __m128 XMM2 = _mm_load_ps((x)+i+ 8); \
        __m128 XMM3 = _mm_load_ps((x)+i+12); \
        XMM0 = _mm_xor_ps(XMM0, XMM4); \
        XMM1 = _mm_xor_ps(XMM1, XMM4); \
        XMM2 = _mm_xor_ps(XMM2, XMM4); \
        XMM3 = _mm_xor_ps(XMM3, XMM4); \
        _mm_store_ps((y)+i   , XMM0); \
        _mm_store_ps((y)+i+ 4, XMM1); \
        _mm_store_ps((y)+i+ 8, XMM2); \
        _mm_store_ps((y)+i+12, XMM3); \
    } \
}

#define vecadd(y, x, c, n) \
{ \
    int i; \
    __m128 XMM7 = _mm_set_ps1(c); \
    for (i = 0;i < (n);i += 8) { \
        __m128 XMM0 = _mm_load_ps((x)+i  ); \
        __m128 XMM1 = _mm_load_ps((x)+i+4); \
        __m128 XMM2 = _mm_load_ps((y)+i  ); \
        __m128 XMM3 = _mm_load_ps((y)+i+4); \
        XMM0 = _mm_mul_ps(XMM0, XMM7); \
        XMM1 = _mm_mul_ps(XMM1, XMM7); \
        XMM2 = _mm_add_ps(XMM2, XMM0); \
        XMM3 = _mm_add_ps(XMM3, XMM1); \
        _mm_store_ps((y)+i  , XMM2); \
        _mm_store_ps((y)+i+4, XMM3); \
    } \
}

#define vecdiff(z, x, y, n) \
{ \
    int i; \
    for (i = 0;i < (n);i += 16) { \
        __m128 XMM0 = _mm_load_ps((x)+i   ); \
        __m128 XMM1 = _mm_load_ps((x)+i+ 4); \
        __m128 XMM2 = _mm_load_ps((x)+i+ 8); \
        __m128 XMM3 = _mm_load_ps((x)+i+12); \
        __m128 XMM4 = _mm_load_ps((y)+i   ); \
        __m128 XMM5 = _mm_load_ps((y)+i+ 4); \
        __m128 XMM6 = _mm_load_ps((y)+i+ 8); \
        __m128 XMM7 = _mm_load_ps((y)+i+12); \
        XMM0 = _mm_sub_ps(XMM0, XMM4); \
        XMM1 = _mm_sub_ps(XMM1, XMM5); \
        XMM2 = _mm_sub_ps(XMM2, XMM6); \
        XMM3 = _mm_sub_ps(XMM3, XMM7); \
        _mm_store_ps((z)+i   , XMM0); \
        _mm_store_ps((z)+i+ 4, XMM1); \
        _mm_store_ps((z)+i+ 8, XMM2); \
        _mm_store_ps((z)+i+12, XMM3); \
    } \
}

#define vecscale(y, c, n) \
{ \
    int i; \
    __m128 XMM7 = _mm_set_ps1(c); \
    for (i = 0;i < (n);i += 8) { \
        __m128 XMM0 = _mm_load_ps((y)+i  ); \
        __m128 XMM1 = _mm_load_ps((y)+i+4); \
        XMM0 = _mm_mul_ps(XMM0, XMM7); \
        XMM1 = _mm_mul_ps(XMM1, XMM7); \
        _mm_store_ps((y)+i  , XMM0); \
        _mm_store_ps((y)+i+4, XMM1); \
    } \
}

#define vecmul(y, x, n) \
{ \
    int i; \
    for (i = 0;i < (n);i += 16) { \
        __m128 XMM0 = _mm_load_ps((x)+i   ); \
        __m128 XMM1 = _mm_load_ps((x)+i+ 4); \
        __m128 XMM2 = _mm_load_ps((x)+i+ 8); \
        __m128 XMM3 = _mm_load_ps((x)+i+12); \
        __m128 XMM4 = _mm_load_ps((y)+i   ); \
        __m128 XMM5 = _mm_load_ps((y)+i+ 4); \
        __m128 XMM6 = _mm_load_ps((y)+i+ 8); \
        __m128 XMM7 = _mm_load_ps((y)+i+12); \
        XMM4 = _mm_mul_ps(XMM4, XMM0); \
        XMM5 = _mm_mul_ps(XMM5, XMM1); \
        XMM6 = _mm_mul_ps(XMM6, XMM2); \
        XMM7 = _mm_mul_ps(XMM7, XMM3); \
        _mm_store_ps((y)+i   , XMM4); \
        _mm_store_ps((y)+i+ 4, XMM5); \
        _mm_store_ps((y)+i+ 8, XMM6); \
        _mm_store_ps((y)+i+12, XMM7); \
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
    __m128 XMM0 = _mm_setzero_ps(); \
    __m128 XMM1 = _mm_setzero_ps(); \
    __m128 XMM2, XMM3, XMM4, XMM5; \
    for (i = 0;i < (n);i += 8) { \
        XMM2 = _mm_load_ps((x)+i  ); \
        XMM3 = _mm_load_ps((x)+i+4); \
        XMM4 = _mm_load_ps((y)+i  ); \
        XMM5 = _mm_load_ps((y)+i+4); \
        XMM2 = _mm_mul_ps(XMM2, XMM4); \
        XMM3 = _mm_mul_ps(XMM3, XMM5); \
        XMM0 = _mm_add_ps(XMM0, XMM2); \
        XMM1 = _mm_add_ps(XMM1, XMM3); \
    } \
    XMM0 = _mm_add_ps(XMM0, XMM1); \
    __horizontal_sum(XMM0, XMM1); \
    _mm_store_ss((s), XMM0); \
}

#define vec2norm(s, x, n) \
{ \
    int i; \
    __m128 XMM0 = _mm_setzero_ps(); \
    __m128 XMM1 = _mm_setzero_ps(); \
    __m128 XMM2, XMM3; \
    for (i = 0;i < (n);i += 8) { \
        XMM2 = _mm_load_ps((x)+i  ); \
        XMM3 = _mm_load_ps((x)+i+4); \
        XMM2 = _mm_mul_ps(XMM2, XMM2); \
        XMM3 = _mm_mul_ps(XMM3, XMM3); \
        XMM0 = _mm_add_ps(XMM0, XMM2); \
        XMM1 = _mm_add_ps(XMM1, XMM3); \
    } \
    XMM0 = _mm_add_ps(XMM0, XMM1); \
    __horizontal_sum(XMM0, XMM1); \
    XMM2 = XMM0; \
    XMM1 = _mm_rsqrt_ss(XMM0); \
    XMM3 = XMM1; \
    XMM1 = _mm_mul_ss(XMM1, XMM1); \
    XMM1 = _mm_mul_ss(XMM1, XMM3); \
    XMM1 = _mm_mul_ss(XMM1, XMM0); \
    XMM1 = _mm_mul_ss(XMM1, _mm_set_ss(-0.5f)); \
    XMM3 = _mm_mul_ss(XMM3, _mm_set_ss(1.5f)); \
    XMM3 = _mm_add_ss(XMM3, XMM1); \
    XMM3 = _mm_mul_ss(XMM3, XMM2); \
    _mm_store_ss((s), XMM3); \
}

#define vec2norminv(s, x, n) \
{ \
    int i; \
    __m128 XMM0 = _mm_setzero_ps(); \
    __m128 XMM1 = _mm_setzero_ps(); \
    __m128 XMM2, XMM3; \
    for (i = 0;i < (n);i += 16) { \
        XMM2 = _mm_load_ps((x)+i  ); \
        XMM3 = _mm_load_ps((x)+i+4); \
        XMM2 = _mm_mul_ps(XMM2, XMM2); \
        XMM3 = _mm_mul_ps(XMM3, XMM3); \
        XMM0 = _mm_add_ps(XMM0, XMM2); \
        XMM1 = _mm_add_ps(XMM1, XMM3); \
    } \
    XMM0 = _mm_add_ps(XMM0, XMM1); \
    __horizontal_sum(XMM0, XMM1); \
    XMM2 = XMM0; \
    XMM1 = _mm_rsqrt_ss(XMM0); \
    XMM3 = XMM1; \
    XMM1 = _mm_mul_ss(XMM1, XMM1); \
    XMM1 = _mm_mul_ss(XMM1, XMM3); \
    XMM1 = _mm_mul_ss(XMM1, XMM0); \
    XMM1 = _mm_mul_ss(XMM1, _mm_set_ss(-0.5f)); \
    XMM3 = _mm_mul_ss(XMM3, _mm_set_ss(1.5f)); \
    XMM3 = _mm_add_ss(XMM3, XMM1); \
    _mm_store_ss((s), XMM3); \
}
