/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_xcopy.h"
#include "libxsmm_main.h"

#if !defined(LIBXSMM_ITRANS_BUFFER_MAXSIZE)
# if defined(NDEBUG)
#   define LIBXSMM_ITRANS_BUFFER_MAXSIZE (12 << 10/*12kB*/)
# else
#   define LIBXSMM_ITRANS_BUFFER_MAXSIZE 1
# endif
#endif
#if !defined(LIBXSMM_MCOPY_JIT_TINY) && 0
# define LIBXSMM_MCOPY_JIT_TINY
#endif


/* definition of corresponding variables */
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT))
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_xcopy_jit);
#endif
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_xcopy_taskscale);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_mcopy_prefetch);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_mcopy_mbytes);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_mzero_mbytes);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_tcopy_mbytes);
LIBXSMM_APIVAR_PUBLIC_DEF(float libxsmm_mcopy_nscale);
LIBXSMM_APIVAR_PUBLIC_DEF(float libxsmm_mzero_nscale);
LIBXSMM_APIVAR_PUBLIC_DEF(float libxsmm_tcopy_nscale);


LIBXSMM_API_INTERN void libxsmm_xcopy_init(int archid)
{
  { /* setup tile sizes according to CPUID or environment */
    if (LIBXSMM_X86_AVX512_CORE <= archid) { /* avx-512/core */
      libxsmm_mcopy_prefetch = 0;
      libxsmm_mcopy_mbytes = 0;
      libxsmm_mcopy_nscale = 0.f;
      libxsmm_mzero_mbytes = 0;
      libxsmm_mzero_nscale = 0.f;
      libxsmm_tcopy_mbytes = 32768;
      libxsmm_tcopy_nscale = 0.f;
    }
    else if (LIBXSMM_X86_AVX512_MIC <= archid && LIBXSMM_X86_AVX512_CORE > archid) {
      libxsmm_mcopy_prefetch = 1;
      libxsmm_mcopy_mbytes = 0;
      libxsmm_mcopy_nscale = 0.f;
      libxsmm_mzero_mbytes = 0;
      libxsmm_mzero_nscale = 0.f;
      libxsmm_tcopy_mbytes = 32768;
      libxsmm_tcopy_nscale = 0.f;
    }
    else { /* avx2 */
      libxsmm_mcopy_prefetch = 0;
      libxsmm_mcopy_mbytes = 0;
      libxsmm_mcopy_nscale = 0.f;
      libxsmm_mzero_mbytes = 8192;
      libxsmm_mzero_nscale = 0.f;
      libxsmm_tcopy_mbytes = 4096;
      libxsmm_tcopy_nscale = 0.f;
    }
  }
  { /* mcopy: load/adjust tile sizes (measured as if DP) */
    const char* const env_m = getenv("LIBXSMM_MCOPY_M"), * const env_n = getenv("LIBXSMM_MCOPY_N");
    const int m = ((NULL == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((NULL == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    if (0 < m) libxsmm_mcopy_mbytes = LIBXSMM_MAX(m, 1) * 8/*DP*/;
    if (0 != libxsmm_mcopy_mbytes && 0 != libxsmm_mcopy_nscale) {
      if (0 < n) libxsmm_mcopy_nscale = ((float)(n * 8/*DP*/)) / libxsmm_mcopy_mbytes;
      if (1 > (libxsmm_mcopy_nscale * libxsmm_mcopy_mbytes)) {
        const float stretch = 1.f / libxsmm_mcopy_mbytes;
        libxsmm_mcopy_nscale = LIBXSMM_MAX(stretch, libxsmm_mcopy_nscale);
      }
    }
  }
  { /* mzero: load/adjust tile sizes (measured as if DP) */
    const char* const env_m = getenv("LIBXSMM_MZERO_M"), * const env_n = getenv("LIBXSMM_MZERO_N");
    const int m = ((NULL == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((NULL == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    if (0 < m) libxsmm_mzero_mbytes = LIBXSMM_MAX(m, 1) * 8/*DP*/;
    if (0 != libxsmm_mzero_mbytes && 0 != libxsmm_mzero_nscale) {
      if (0 < n) libxsmm_mzero_nscale = ((float)(n * 8/*DP*/)) / libxsmm_mzero_mbytes;
      if (1 > (libxsmm_mzero_nscale * libxsmm_mzero_mbytes)) {
        const float stretch = 1.f / libxsmm_mzero_mbytes;
        libxsmm_mzero_nscale = LIBXSMM_MAX(stretch, libxsmm_mzero_nscale);
      }
    }
  }
  { /* tcopy: load/adjust tile sizes (measured as if DP) */
    const char* const env_m = getenv("LIBXSMM_TCOPY_M"), * const env_n = getenv("LIBXSMM_TCOPY_N");
    const int m = ((NULL == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((NULL == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    if (0 < m) libxsmm_tcopy_mbytes = LIBXSMM_MAX(m, 1) * 8/*DP*/;
    if (0 != libxsmm_tcopy_mbytes && 0 != libxsmm_tcopy_nscale) {
      if (0 < n) libxsmm_tcopy_nscale = ((float)(n * 8/*DP*/)) / libxsmm_tcopy_mbytes;
      if (1 > (libxsmm_tcopy_nscale * libxsmm_tcopy_mbytes)) {
        const float stretch = 1.f / libxsmm_tcopy_mbytes;
        libxsmm_tcopy_nscale = LIBXSMM_MAX(stretch, libxsmm_tcopy_nscale);
      }
    }
  }
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT))
  { /* check if JIT-code generation is permitted */
    const char *const env_jit = getenv("LIBXSMM_XCOPY_JIT");
    libxsmm_xcopy_jit = ((NULL == env_jit || 0 == *env_jit) ? (LIBXSMM_XCOPY_JIT) : atoi(env_jit));
# if defined(LIBXSMM_XCOPY_MELTW)
    if (LIBXSMM_X86_AVX512_CORE > archid) libxsmm_xcopy_jit &= ~2;
# endif
  }
#endif
  { /* determines if OpenMP tasks are used (when available) */
    const char *const env_t = getenv("LIBXSMM_XCOPY_TASKS");
    libxsmm_xcopy_taskscale = ((NULL == env_t || 0 == *env_t)
      ? 0/*disabled*/ : (LIBXSMM_XCOPY_TASKSCALE * atoi(env_t)));
  }
}


LIBXSMM_API_INTERN void libxsmm_xcopy_finalize(void)
{
}


LIBXSMM_API void libxsmm_matcopy_task_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int km, unsigned int kn, libxsmm_xcopykernel kernel,
  int tid, int ntasks)
{
  const unsigned int tm = (0 == km ? m : km);
  const unsigned int tn = (0 == kn ? LIBXSMM_MIN(LIBXSMM_XCOPY_TILE_MIN, n) : kn);
  const int mtasks = LIBXSMM_UPDIV(m, tm);
  unsigned int m0, m1, n0, n1;

  LIBXSMM_ASSERT_MSG(tid < ntasks && 0 < ntasks, "Invalid task setup");
  LIBXSMM_ASSERT_MSG(tm <= m && tn <= n, "Invalid problem size");
  LIBXSMM_ASSERT_MSG(0 < tm && 0 < tn, "Invalid tile size");
  LIBXSMM_ASSERT_MSG(typesize <= 255, "Invalid type-size");
  LIBXSMM_ASSERT(0 < mtasks);

  if (ntasks <= mtasks) { /* parallelized over M */
    const unsigned int mt = LIBXSMM_UPDIV(m, ntasks);
    m0 = LIBXSMM_MIN(tid * mt, m);
    m1 = LIBXSMM_MIN(m0 + mt, m);
    n0 = 0; n1 = n;
  }
  else { /* parallelized over M and N */
    const int mntasks = ntasks / mtasks;
    const int mtid = tid / mntasks, ntid = tid - mtid * mntasks;
    const unsigned int nt = LIBXSMM_UP(LIBXSMM_UPDIV(n, mntasks), tn) ;
    m0 = LIBXSMM_MIN(mtid * tm, m); m1 = LIBXSMM_MIN(m0 + tm, m);
    n0 = LIBXSMM_MIN(ntid * nt, n); n1 = LIBXSMM_MIN(n0 + nt, n);
  }

  LIBXSMM_ASSERT_MSG(m0 <= m1 && m1 <= m, "Invalid task size");
  LIBXSMM_ASSERT_MSG(n0 <= n1 && n1 <= n, "Invalid task size");

  if (NULL != in) { /* copy-kernel */
    libxsmm_matcopy_internal(out, in, typesize, ldi, ldo,
      m0, m1, n0, n1, tm, tn, kernel);
  }
  else {
    libxsmm_matzero_internal(out, typesize, ldo,
      m0, m1, n0, n1, tm, tn, kernel);
  }
}


LIBXSMM_API void libxsmm_otrans_task_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int km, unsigned int kn, libxsmm_xcopykernel kernel,
  int tid, int ntasks)
{
  const unsigned int tm = (0 == km ? m : km);
  const unsigned int tn = (0 == kn ? LIBXSMM_MIN(LIBXSMM_XCOPY_TILE_MIN, n) : kn);
  const int mtasks = LIBXSMM_UPDIV(m, tm);
  unsigned int m0, m1, n0, n1;

  LIBXSMM_ASSERT_MSG(tid < ntasks && 0 < ntasks, "Invalid task setup");
  LIBXSMM_ASSERT_MSG(tm <= m && tn <= n, "Invalid problem size");
  LIBXSMM_ASSERT_MSG(0 < tm && 0 < tn, "Invalid tile size");
  LIBXSMM_ASSERT_MSG(typesize <= 255, "Invalid type-size");
  LIBXSMM_ASSERT(0 < mtasks);

  if (ntasks <= mtasks) { /* parallelized over M */
    const unsigned int mt = LIBXSMM_UPDIV(m, ntasks);
    m0 = LIBXSMM_MIN(tid * mt, m);
    m1 = LIBXSMM_MIN(m0 + mt, m);
    n0 = 0; n1 = n;
  }
  else { /* parallelized over M and N */
    const int mntasks = ntasks / mtasks;
    const int mtid = tid / mntasks, ntid = tid - mtid * mntasks;
    const unsigned int nt = LIBXSMM_UP(LIBXSMM_UPDIV(n, mntasks), tn);
    m0 = LIBXSMM_MIN(mtid * tm, m); m1 = LIBXSMM_MIN(m0 + tm, m);
    n0 = LIBXSMM_MIN(ntid * nt, n); n1 = LIBXSMM_MIN(n0 + nt, n);
  }

  LIBXSMM_ASSERT_MSG(m0 <= m1 && m1 <= m, "Invalid task size");
  LIBXSMM_ASSERT_MSG(n0 <= n1 && n1 <= n, "Invalid task size");

  libxsmm_otrans_internal(out, in, typesize, ldi, ldo, m0, m1, n0, n1, tm, tn, kernel);
}


LIBXSMM_API_INTERN void libxsmm_matcopy_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xcopykernel kernel)
{
  LIBXSMM_ASSERT(NULL != in);
  if (NULL != kernel.ptr) {
    const libxsmm_descriptor* desc;
    libxsmm_code_pointer code;
    code.ptr_const = kernel.ptr;
    LIBXSMM_EXPECT_NOT(NULL, libxsmm_get_kernel_xinfo(code, &desc, NULL/*code_size*/));
    LIBXSMM_ASSERT(NULL != desc);
#if defined(LIBXSMM_XCOPY_MELTW)
    LIBXSMM_ASSERT(LIBXSMM_KERNEL_KIND_MELTW == LIBXSMM_DESCRIPTOR_KIND(desc->kind));
#else
    LIBXSMM_ASSERT(LIBXSMM_KERNEL_KIND_MCOPY == LIBXSMM_DESCRIPTOR_KIND(desc->kind));
    if (0 != desc->mcopy.desc.prefetch) {
      LIBXSMM_XCOPY(LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL_PF, kernel,
        out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1);
      return;
    }
#endif
  }
  LIBXSMM_XCOPY(LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL, kernel,
    out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1);
}


LIBXSMM_API_INTERN void libxsmm_matzero_internal(void* out, unsigned int typesize, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xcopykernel kernel)
{
  /* coverity[ptr_arith] */
  LIBXSMM_XCOPY(LIBXSMM_MZERO_KERNEL, LIBXSMM_MZERO_CALL, kernel,
    out, NULL, typesize, 0, ldo, tm, tn, m0, m1, n0, n1);
}


LIBXSMM_API_INTERN void libxsmm_otrans_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xcopykernel kernel)
{
  LIBXSMM_ASSERT(NULL != in);
  LIBXSMM_XCOPY(LIBXSMM_TCOPY_KERNEL, LIBXSMM_TCOPY_CALL, kernel,
    out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1);
}


LIBXSMM_API void libxsmm_matcopy_task(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  int tid, int ntasks)
{
  LIBXSMM_INIT
  if (0 < typesize && 256 > typesize && m <= ldi && m <= ldo && out != in &&
    ((NULL != out && 0 < m && 0 < n) || (0 == m && 0 == n)) &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < ntasks)
  {
    if (0 < m && 0 < n) {
      unsigned int tm, tn, ts;
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT)) && !defined(LIBXSMM_XCOPY_MELTW)
      int prefetch = 0;
#endif
      libxsmm_xcopykernel kernel;
      kernel.ptr = NULL;
      if (NULL != in) { /* mcopy */
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT)) && !defined(LIBXSMM_XCOPY_MELTW)
        prefetch = libxsmm_mcopy_prefetch;
#endif
        tm = LIBXSMM_UPDIV(libxsmm_mcopy_mbytes, typesize);
        tn = (unsigned int)(libxsmm_mcopy_nscale * tm);
        ts = libxsmm_mcopy_mbytes;
      }
      else { /* mzero */
        tm = LIBXSMM_UPDIV(libxsmm_mzero_mbytes, typesize);
        tn = (unsigned int)(libxsmm_mzero_nscale * tm);
        ts = libxsmm_mzero_mbytes;
      }
      if (0 == tm) tm = m;
      if (0 == tn) tn = LIBXSMM_MIN(LIBXSMM_XCOPY_TILE_MIN, n);
      if (0 != ts && ts < (tm * tn * typesize)) {
        tm = LIBXSMM_MAX(ts / (tn * typesize), LIBXSMM_XCOPY_TILE_MIN);
      }
      if ((unsigned int)m < tm || (unsigned int)n < tn) {
        if (1 == ntasks) {
          tm = (unsigned int)m; tn = (unsigned int)n;
        }
        else {
          const unsigned int tasksize = (((unsigned int)m) * (unsigned int)n) / ((unsigned int)(ntasks * libxsmm_mcopy_nscale));
          const unsigned int nn = libxsmm_isqrt_u32(tasksize);
          const unsigned int mm = (unsigned int)(libxsmm_mcopy_nscale * nn);
          tn = LIBXSMM_CLMP((unsigned int)n, 1, nn);
          tm = LIBXSMM_CLMP((unsigned int)m, 1, mm);
        }
      }
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT))
# if !defined(LIBXSMM_MCOPY_JIT_TINY)
      else
# endif
      if (0 != (2 & libxsmm_xcopy_jit)) { /* JIT'ted matrix-copy permitted? */
# if defined(LIBXSMM_XCOPY_MELTW)
        const libxsmm_blasint sldi = ldi * typesize, sldo = ldo * typesize;
        if (NULL != in) { /* mcopy */
          kernel.meltw_copy = libxsmm_dispatch_meltw_copy(
            (libxsmm_blasint)tm * typesize, (libxsmm_blasint)tn * typesize,
            &sldi, &sldo, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8);
        }
        else { /* mzero */
          kernel.meltw_zero = libxsmm_dispatch_meltw_zero(
            (libxsmm_blasint)tm * typesize, (libxsmm_blasint)tn * typesize,
            &sldi, &sldo, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8);
        }
# else
        libxsmm_descriptor_blob blob;
        kernel.xmcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&blob,
          typesize, tm, tn, (unsigned int)ldo, (unsigned int)ldi,
          NULL != in ? LIBXSMM_MATCOPY_FLAG_DEFAULT : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE,
          prefetch, NULL/*default unroll*/));
# endif
      }
#endif
      libxsmm_matcopy_task_internal(out, in, typesize,
        (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
        tm, tn, kernel, tid, ntasks);
    }
  }
  else {
    static int error_once = 0;
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 > tid || tid >= ntasks) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix-copy thread-id or number of tasks is incorrect!\n");
      }
      else if (NULL == out) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix-copy input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the matrix-copy must be different!\n");
      }
      else if (0 == typesize || 256 <= typesize) {
        fprintf(stderr, "LIBXSMM ERROR: invalid type-size for matrix-copy specified!\n");
      }
      else if (ldi < m || ldo < m) {
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the matrix-copy is/are too small!\n");
      }
      else if (0 > m || 0 > n) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix extent(s) of the matrix-copy is/are negative!\n");
      }
    }
  }
}


LIBXSMM_API void libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  libxsmm_matcopy_task(out, in, typesize, m, n, ldi, ldo, 0/*tid*/, 1/*ntasks*/);
}


LIBXSMM_API void libxsmm_otrans_task(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  int tid, int ntasks)
{
  static int error_once = 0;
  LIBXSMM_INIT
  if (0 < typesize && 256 > typesize && m <= ldi && n <= ldo &&
    ((NULL != out && NULL != in && 0 < m && 0 < n) || (0 == m && 0 == n)) &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < ntasks)
  {
    if (0 < m && 0 < n) {
      if (out != in) {
        unsigned int tm = LIBXSMM_UPDIV(libxsmm_tcopy_mbytes, typesize);
        unsigned int tn = (unsigned int)(libxsmm_tcopy_nscale * tm);
        libxsmm_xcopykernel kernel;
        kernel.ptr = NULL;
        if (0 == tm) tm = m;
        if (0 == tn) tn = LIBXSMM_MIN(LIBXSMM_XCOPY_TILE_MIN, n);
        if (0 != libxsmm_tcopy_mbytes && libxsmm_tcopy_mbytes < (tm * tn * typesize)) {
          tm = LIBXSMM_MAX(libxsmm_tcopy_mbytes / (tn * typesize), LIBXSMM_XCOPY_TILE_MIN);
        }
        if ((unsigned int)m < tm || (unsigned int)n < tn) {
          if (1 == ntasks) {
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT))
            libxsmm_descriptor_blob blob;
            if (0 != (1 & libxsmm_xcopy_jit) /* JIT'ted transpose permitted? */
              && NULL != (kernel.xtrans = libxsmm_dispatch_trans( /* JIT-kernel available? */
                libxsmm_trans_descriptor_init(&blob, typesize, (unsigned int)m, (unsigned int)n, (unsigned int)ldo))))
            {
              LIBXSMM_TCOPY_CALL(kernel, typesize, in, ldi, out, ldo);
              return; /* fast path */
            }
            LIBXSMM_ASSERT(NULL == kernel.ptr);
#endif
            tm = (unsigned int)m; tn = (unsigned int)n;
          }
          else {
            const unsigned int tasksize = (((unsigned int)m) * (unsigned int)n) / ((unsigned int)(ntasks * libxsmm_tcopy_nscale));
            const unsigned int nn = libxsmm_isqrt_u32(tasksize);
            const unsigned int mm = (unsigned int)(libxsmm_tcopy_nscale * nn);
            tn = LIBXSMM_CLMP((unsigned int)n, 1, nn);
            tm = LIBXSMM_CLMP((unsigned int)m, 1, mm);
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT))
            { const libxsmm_trans_descriptor* desc;
              libxsmm_descriptor_blob blob;
              if (0 != (1 & libxsmm_xcopy_jit) /* JIT'ted transpose permitted? */
                && NULL != (desc = libxsmm_trans_descriptor_init(&blob, typesize, tm, tn, (unsigned int)ldo)))
              {
                kernel.xtrans = libxsmm_dispatch_trans(desc);
              }
            }
#endif
          }
        }
        libxsmm_otrans_task_internal(out, in, typesize,
          (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
          tm, tn, kernel, tid, ntasks);
      }
      else if (ldi == ldo) {
        libxsmm_itrans(out, typesize, m, n, ldi);
      }
      else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the transpose must be different!\n");
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 > tid || tid >= ntasks) {
        fprintf(stderr, "LIBXSMM ERROR: the transpose thread-id or number of tasks is incorrect!\n");
      }
      else if (NULL == out || NULL == in) {
        fprintf(stderr, "LIBXSMM ERROR: the transpose input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the transpose must be different!\n");
      }
      else if (0 == typesize || 256 <= typesize) {
        fprintf(stderr, "LIBXSMM ERROR: invalid type-size for matrix-transpose specified!\n");
      }
      else if (ldi < m || ldo < n) {
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the transpose is/are too small!\n");
      }
      else if (0 > m || 0 > n) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix extent(s) of the transpose is/are negative!\n");
      }
    }
  }
}


LIBXSMM_API void libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  libxsmm_otrans_task(out, in, typesize, m, n, ldi, ldo, 0/*tid*/, 1/*ntasks*/);
}


LIBXSMM_API_INTERN void libxsmm_itrans_internal(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  const signed char c = (signed char)typesize;
  libxsmm_blasint i, j;
  LIBXSMM_ASSERT(NULL != inout && 0 < typesize && m <= ld && n <= ld);
  LIBXSMM_ASSERT(m == n && typesize <= 127);
  for (i = 0; i < m; ++i) {
    for (j = 0; j < i; ++j) {
      char *const a = &((char*)inout)[(i*ld+j)*typesize];
      char *const b = &((char*)inout)[(j*ld+i)*typesize];
      signed char k = 0;
      for (; k < c; ++k) LIBXSMM_ISWAP(a[k], b[k]);
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_itrans_scratch(void* inout, void* scratch, unsigned int typesize,
  size_t size, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  LIBXSMM_ASSERT(NULL != inout && 0 < typesize && m <= ld && n <= ld);
  memcpy(scratch, inout, size);
  libxsmm_otrans(inout, scratch, typesize, m, n, ld, ld);
}


LIBXSMM_API void libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  static int error_once = 0;
  if (NULL != inout && 0 < typesize && m <= ld && n <= ld) {
    if (m == n && typesize <= 127) {
      libxsmm_itrans_internal(inout, typesize, m, n, ld);
    }
    else {
      /* TODO: reduce memory consumption by relying on matcopy */
      const size_t size = ld * LIBXSMM_MAX(m, n) * typesize;
      if (size <= LIBXSMM_ITRANS_BUFFER_MAXSIZE) {
        char buffer[LIBXSMM_ITRANS_BUFFER_MAXSIZE];
        libxsmm_itrans_scratch(inout, buffer, typesize, size, m, n, ld);
      }
      else {
        void* buffer;
        LIBXSMM_INIT
        if (EXIT_SUCCESS == libxsmm_xmalloc(&buffer, size, 0/*auto-align*/,
          LIBXSMM_MALLOC_FLAG_SCRATCH | LIBXSMM_MALLOC_FLAG_PRIVATE,
          0/*extra*/, 0/*extra_size*/))
        {
          LIBXSMM_ASSERT(NULL != buffer);
          libxsmm_itrans_scratch(inout, buffer, typesize, size, m, n, ld);
          libxsmm_xfree(buffer, 0/*no check*/);
        }
        else if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: failed to allocate buffer for in-place transpose!\n");
        }
      }
    }
  }
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid argument(s) for in-place transpose!\n");
  }
}


LIBXSMM_API void libxsmm_itrans_batch(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride[], libxsmm_blasint batchsize,
  /*unsigned*/int tid, /*unsigned*/int ntasks)
{
  static int error_once = 0;
  if (NULL != inout && 0 < typesize && m <= ld && n <= ld) {
    /* TODO: reduce memory consumption by relying on matcopy */
    const size_t scratchsize = ld * LIBXSMM_MAX(m, n) * typesize;
    const libxsmm_blasint size = LIBXSMM_ABS(batchsize);
    const libxsmm_blasint tasksize = LIBXSMM_UPDIV(size, ntasks);
    const libxsmm_blasint begin = tid * tasksize, span = begin + tasksize;
    const libxsmm_blasint end = LIBXSMM_MIN(span, size);
    char buffer[LIBXSMM_ITRANS_BUFFER_MAXSIZE];
    char *const mat0 = (char*)inout;
    void* scratch = NULL;
    if (m != n || 127 < typesize) {
      if (scratchsize <= LIBXSMM_ITRANS_BUFFER_MAXSIZE) {
        scratch = buffer;
      }
      else {
        LIBXSMM_INIT
        if (EXIT_SUCCESS != libxsmm_xmalloc(&scratch, scratchsize, 0/*auto-align*/,
            LIBXSMM_MALLOC_FLAG_SCRATCH | LIBXSMM_MALLOC_FLAG_PRIVATE,
            0/*extra*/, 0/*extra_size*/)
          && 0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: failed to allocate buffer for in-place transpose!\n");
        }
      }
    }
    if (NULL != stride) {
      if (0 != index_stride) { /* stride array contains indexes */
        libxsmm_blasint i;
        if (NULL == scratch) {
          for (i = begin * index_stride; i < (end * index_stride); i += index_stride) {
            char *const mat = &mat0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride, i) - index_base) * typesize];
            libxsmm_itrans_internal(mat, typesize, m, n, ld);
          }
        }
        else {
          for (i = begin * index_stride; i < (end * index_stride); i += index_stride) {
            char *const mat = &mat0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride, i) - index_base) * typesize];
            libxsmm_itrans_scratch(mat, scratch, typesize, scratchsize, m, n, ld);
          }
        }
      }
      else { /* singular stride is measured in Bytes */
        const libxsmm_blasint d = *stride - index_base * sizeof(void*);
        const char *const endi = mat0 + end * d;
        char* i = mat0 + begin * d;
        if (NULL == scratch) {
          for (; i < endi; i += d) {
            void *const mat = *((void**)i);
#if defined(LIBXSMM_BATCH_CHECK)
            if (NULL != mat) libxsmm_itrans_internal(mat, typesize, m, n, ld);
#else
            libxsmm_itrans_internal(mat, typesize, m, n, ld);
#endif
          }
        }
        else {
          for (; i < endi; i += d) {
            void *const mat = *((void**)i);
#if defined(LIBXSMM_BATCH_CHECK)
            if (NULL != mat) libxsmm_itrans_scratch(mat, scratch, typesize, scratchsize, m, n, ld);
#else
            libxsmm_itrans_scratch(mat, scratch, typesize, scratchsize, m, n, ld);
#endif
          }
        }
      }
      if (NULL != scratch && LIBXSMM_ITRANS_BUFFER_MAXSIZE < scratchsize) {
        libxsmm_xfree(scratch, 0/*no check*/);
      }
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: invalid stride for in-place batch-transpose!\n");
    }
  }
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid argument(s) for in-place batch-transpose!\n");
  }
}


#if defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matcopy)(void* /*out*/, const void* /*in*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matcopy)(void* out, const void* in, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  ldx = *(NULL != ldi ? ldi : m);
  libxsmm_matcopy(out, in, (unsigned int)*typesize, *m, *(NULL != n ? n : m), ldx, NULL != ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_otrans)(void* /*out*/, const void* /*in*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_otrans)(void* out, const void* in, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  ldx = *(NULL != ldi ? ldi : m);
  libxsmm_otrans(out, in, (unsigned int)*typesize, *m, *(NULL != n ? n : m), ldx, NULL != ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* /*inout*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* inout, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  libxsmm_itrans(inout, (unsigned int)*typesize, *m, *(NULL != n ? n : m), *(NULL != ld ? ld : m));
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

