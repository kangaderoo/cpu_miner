/*
 * Copyright 2013 gerko.deroo@kangaderoo.nl
 * All rights reserved. 
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <immintrin.h>
#include "cpuminer-config.h"
#include "miner.h"
#include <stdlib.h>
#include <inttypes.h>

#if defined(__x86_64__)

static inline void xor_salsa_sidm(__m128i *calc_18, __m128i  *calc_13,__m128i  *calc_9, __m128i *calc_7,
  								  __m128i *calc_1,  __m128i  *calc_4,__m128i  *calc_3, __m128i *calc_2)
{
	int i;
	__m128i _calc;
	__m128i _shift_left;
	__m128i row1;
	__m128i row2;
	__m128i row3;
	__m128i row4;

	*calc_18 = _mm_xor_si128(*calc_18, *calc_1);
	*calc_7 = _mm_xor_si128(*calc_7, *calc_2);
	*calc_9 = _mm_xor_si128(*calc_9, *calc_3);
	*calc_13 = _mm_xor_si128(*calc_13, *calc_4);
	row1 = *calc_18;
	row2 = *calc_7;
	row3 = *calc_3;
	row4 = *calc_13;

	for (i = 0; i < 8; i += 2) {
		/* first row */
 		_calc = _mm_add_epi32(row1, row4);
		_shift_left = _mm_slli_epi32(_calc, 7);
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		row2 = _mm_xor_si128(row2, _calc);
		row2 = _mm_xor_si128(row2, _shift_left);

		/* second row */
		_calc = _mm_add_epi32(row2, row1);
		_shift_left = _mm_slli_epi32(_calc, 9);
		_calc = _mm_srli_epi32(_calc,(32 - 9));
		row3 = _mm_xor_si128(row3, _calc);
		row3 = _mm_xor_si128(row3, _shift_left);

		/* third row */
		_calc = _mm_add_epi32(row3, row2);
		_shift_left = _mm_slli_epi32(_calc, 13);
		_calc = _mm_srli_epi32(_calc,(32 - 13));
		row4 = _mm_xor_si128(row4, _calc);
		row4 = _mm_xor_si128(row4, _shift_left);

		/* fourth row */
		_calc = _mm_add_epi32(row4, row3);
		_shift_left = _mm_slli_epi32(_calc, 18);
		_calc = _mm_srli_epi32(_calc,(32 - 18));
		row1 = _mm_xor_si128(row1, _calc);
		row1 = _mm_xor_si128(row1, _shift_left);

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2 = _mm_shuffle_epi32(row2,0x93);
		row3 = _mm_shuffle_epi32(row3,0x4e);
		row4 = _mm_shuffle_epi32(row4,0x39);
	// end transpose

		// switch *calc_13 and * calc_7 usage compared to rows
		/* first column */
		_calc = _mm_add_epi32(row1, row2);
		_shift_left = _mm_slli_epi32(_calc, 7);
		_calc = _mm_srli_epi32(_calc,(32 - 7));
		row4 = _mm_xor_si128(row4, _calc);
		row4 = _mm_xor_si128(row4, _shift_left);

		/* second column */
		_calc = _mm_add_epi32(row4, row1);
		_shift_left = _mm_slli_epi32(_calc, 9);
		_calc = _mm_srli_epi32(_calc,(32 - 9));
		row3 = _mm_xor_si128(row3, _calc);
		row3 = _mm_xor_si128(row3, _shift_left);

		/* third column */
		_calc = _mm_add_epi32(row3, row4);
		_shift_left = _mm_slli_epi32(_calc, 13);
		_calc = _mm_srli_epi32(_calc,(32 - 13));
		row2 = _mm_xor_si128(row2, _calc);
		row2 = _mm_xor_si128(row2, _shift_left);

		/* fourth column */
		_calc = _mm_add_epi32(row2, row3);
		_shift_left = _mm_slli_epi32(_calc, 18);
		_calc = _mm_srli_epi32(_calc,(32 - 18));
		row1 = _mm_xor_si128(row1, _calc);
		row1 = _mm_xor_si128(row1, _shift_left);

	// transpose_matrix(row1, row2, row3, row4, row_to_column);
		row2 = _mm_shuffle_epi32(row2,0x39);
		row3 = _mm_shuffle_epi32(row3,0x4e);
		row4 = _mm_shuffle_epi32(row4,0x93);
	// end transpose
	}
	*calc_18 = _mm_add_epi32(*calc_18,row1);
	*calc_7 = _mm_add_epi32(*calc_7, row2);
	*calc_9 = _mm_add_epi32(*calc_9, row3);
	*calc_13 = _mm_add_epi32(*calc_13, row4);
}

static inline void scrypt_core_sidm(__m128i *X /*, uint32_t *V*/)
{
	uint32_t i, j;
	static __m128i scratch[1025 * 8];

	__m128i *calc_1 = (__m128i*) &X[0];
	__m128i *calc_2 = (__m128i*) &X[1];
	__m128i *calc_3 = (__m128i*) &X[2];
	__m128i *calc_4 = (__m128i*) &X[3];

	__m128i *calc_11 = (__m128i*) &X[4];
	__m128i *calc_21 = (__m128i*) &X[5];
	__m128i *calc_31 = (__m128i*) &X[6];
	__m128i *calc_41 = (__m128i*) &X[7];

	__m128i _calc5;
	__m128i _calc6;
	__m128i _calc7;
	__m128i _calc8;

	/* transpose the data from *X */
	_calc5 =_mm_blend_epi16(*calc_11, *calc_31, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_21, *calc_41, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_31, *calc_11, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_41, *calc_21, 0x0f);
	*calc_11 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_21 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_31 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_41 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1, *calc_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2, *calc_4, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3, *calc_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4, *calc_2, 0x0f);
	*calc_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	for (i = 0; i < 1024; i++) {
		scratch[i * 8 + 0] = *calc_1;
		scratch[i * 8 + 1] = *calc_2;
		scratch[i * 8 + 2] = *calc_3;
		scratch[i * 8 + 3] = *calc_4;
		scratch[i * 8 + 4] = *calc_11;
		scratch[i * 8 + 5] = *calc_21;
		scratch[i * 8 + 6] = *calc_31;
		scratch[i * 8 + 7] = *calc_41;

		xor_salsa_sidm( calc_1, calc_2, calc_3, calc_4, calc_11, calc_21, calc_31, calc_41);
		xor_salsa_sidm(calc_11,calc_21,calc_31,calc_41,  calc_1,  calc_2,  calc_3,  calc_4);
	}

	for (i = 0; i < 1024; i++) {
		j = 8 * (_mm_extract_epi16(*calc_11,0x1) & 1023);

		*calc_1 = _mm_xor_si128(*calc_1, scratch[j]);
		*calc_2 = _mm_xor_si128(*calc_2, scratch[j+1]);
		*calc_3 = _mm_xor_si128(*calc_3, scratch[j+2]);
		*calc_4 = _mm_xor_si128(*calc_4, scratch[j+3]);
		*calc_11 = _mm_xor_si128(*calc_11, scratch[j+4]);
		*calc_21 = _mm_xor_si128(*calc_21, scratch[j+5]);
		*calc_31 = _mm_xor_si128(*calc_31, scratch[j+6]);
		*calc_41 = _mm_xor_si128(*calc_41, scratch[j+7]);

		xor_salsa_sidm(calc_1,calc_2,calc_3,calc_4,calc_11,calc_21,calc_31,calc_41);
		xor_salsa_sidm(calc_11,calc_21,calc_31,calc_41,calc_1,calc_2,calc_3,calc_4);
	}

	_calc5 =_mm_blend_epi16(*calc_11, *calc_31, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_21, *calc_41, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_31, *calc_11, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_41, *calc_21, 0x0f);
	*calc_11 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_21 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_31 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_41 = _mm_blend_epi16(_calc8, _calc7, 0xcc);

	_calc5 =_mm_blend_epi16(*calc_1, *calc_3, 0xf0);
	_calc6 =_mm_blend_epi16(*calc_2, *calc_4, 0x0f);
	_calc7 =_mm_blend_epi16(*calc_3, *calc_1, 0xf0);
	_calc8 =_mm_blend_epi16(*calc_4, *calc_2, 0x0f);
	*calc_1 = _mm_blend_epi16(_calc5, _calc8, 0xcc);
	*calc_2 = _mm_blend_epi16(_calc6, _calc5, 0xcc);
	*calc_3 = _mm_blend_epi16(_calc7, _calc6, 0xcc);
	*calc_4 = _mm_blend_epi16(_calc8, _calc7, 0xcc);
}


#endif


