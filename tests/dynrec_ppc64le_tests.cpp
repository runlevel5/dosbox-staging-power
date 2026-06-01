// SPDX-FileCopyrightText:  2020-2026 Cameron Kaiser, Trung Lê
// SPDX-License-Identifier: GPL-2.0-or-later

// Execution tests for the PPC64LE dynamic recompiler code generator
// (src/cpu/core_dynrec/risc_ppc64le.h).
//
// Each test makes the generator emit real PowerPC machine code into an
// executable page, runs it, and compares the result against a reference
// computed in plain C++. The tests therefore only do anything on a ppc64le
// host; on every other architecture the translation unit is empty.
//
// The generator is a set of `static` functions that normally only compile as
// part of core_dynrec.cpp, which first supplies a pile of context (the code
// cache, Segs[]/cpu_regs[]/fpu, get_CF(), the lazy-flag enum, ...). To exercise
// the real header without dragging in the whole CPU core, we provide minimal
// stand-ins for that context and #include the header inside an anonymous
// namespace. Internal linkage keeps our stand-ins from clashing with the real
// symbols that dosboxcommon also defines, and including the real header (rather
// than a copy) means these tests cannot silently drift from the shipped code.

#if defined(__powerpc64__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/auxv.h>
#include <sys/mman.h>
#include <unistd.h>

// risc_ppc64le.h pulls in <cstdlib>/<sys/auxv.h>; including them here at global
// scope first means the header's copies are guarded no-ops inside the anonymous
// namespace below.

namespace {

// --- minimal stand-ins for the context risc_ppc64le.h expects ---------------

using Bits = long;
using Bitu = unsigned long;

// emit cursor used by the IMM_OP/EXT_OP/... macros via cache_addd()
struct CacheCursor {
	uint8_t* pos = nullptr;
};
CacheCursor cache;

inline void cache_addd(uint32_t v)
{
	std::memcpy(cache.pos, &v, sizeof(v));
	cache.pos += sizeof(v);
}

// Globals whose addresses the generator takes. None of the instructions under
// test dereference them at run time, they only need to exist and link.
uint64_t Segs[256];
uint64_t cpu_regs[256];
uint32_t get_CF() { return 0; }

// Suppress the real fpu/fpu.h (the header pulls it in) and supply a stub.
#define DOSBOX_FPU_H
struct FPU_rec {
	uint8_t opaque[1024];
};
FPU_rec fpu;

// x86 flag-changing instruction types. Mirrors the anonymous enum in
// src/cpu/lazyflags.h; gen_fill_function_ptr() switches on these values, so the
// ordering must match the shipped header.
enum {
	t_UNKNOWN = 0,
	t_ADDb, t_ADDw, t_ADDd,
	t_ORb,  t_ORw,  t_ORd,
	t_ADCb, t_ADCw, t_ADCd,
	t_SBBb, t_SBBw, t_SBBd,
	t_ANDb, t_ANDw, t_ANDd,
	t_SUBb, t_SUBw, t_SUBd,
	t_XORb, t_XORw, t_XORd,
	t_CMPb, t_CMPw, t_CMPd,
	t_INCb, t_INCw, t_INCd,
	t_DECb, t_DECw, t_DECd,
	t_TESTb, t_TESTw, t_TESTd,
	t_SHLb, t_SHLw, t_SHLd,
	t_SHRb, t_SHRw, t_SHRd,
	t_SARb, t_SARw, t_SARd,
	t_ROLb, t_ROLw, t_ROLd,
	t_RORb, t_RORw, t_RORd,
	t_RCLb, t_RCLw, t_RCLd,
	t_RCRb, t_RCRw, t_RCRd,
	t_NEGb, t_NEGw, t_NEGd,
	t_DSHLw, t_DSHLd,
	t_DSHRw, t_DSHRd,
	t_MUL, t_DIV,
	t_NOTDONE,
	t_LASTFLAG
};

// The header is a chunk of an otherwise-monolithic TU, so most of its static
// helpers are unused here; silence the resulting warnings only for the include.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "cpu/core_dynrec/risc_ppc64le.h"
#pragma GCC diagnostic pop

// --- executable-page helper -------------------------------------------------

constexpr uint32_t PpcBlr = 0x4e800020; // blr

class JitPage {
public:
	JitPage()
	{
		mem_ = static_cast<uint8_t*>(mmap(nullptr, kSize,
		                                  PROT_READ | PROT_WRITE,
		                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
		EXPECT_NE(mem_, MAP_FAILED);
		cache.pos = mem_;
	}
	~JitPage()
	{
		if (mem_ != MAP_FAILED)
			munmap(mem_, kSize);
	}
	JitPage(const JitPage&)            = delete;
	JitPage& operator=(const JitPage&) = delete;

	uint8_t* base() const { return mem_; }

	// Append `blr`, flush caches, flip the page to read+execute and return it
	// as a callable of type Fn.
	template <typename Fn>
	Fn seal()
	{
		cache_addd(PpcBlr);
		__builtin___clear_cache(reinterpret_cast<char*>(mem_),
		                        reinterpret_cast<char*>(cache.pos));
		EXPECT_EQ(mprotect(mem_, kSize, PROT_READ | PROT_EXEC), 0);
		return reinterpret_cast<Fn>(mem_);
	}

private:
	static constexpr size_t kSize = 4096;
	uint8_t* mem_                  = nullptr;
};

// Emit `gen_and_imm(r3, mask)` and run it on `value`.
uint32_t emit_and_imm(uint32_t value, uint32_t mask)
{
	JitPage page;
	gen_and_imm(HOST_R3, mask);
	return page.seal<uint32_t (*)(uint32_t)>()(value);
}

// Emit `gen_mov_qword_to_reg_imm(r3, imm)` and return the loaded value.
// On POWER10 this exercises the ISA 3.1 pli / pla fast paths; everywhere else
// (and for constants that fit no fast path) the portable lis/ori sequence.
uint64_t emit_mov_qword(uint64_t imm)
{
	JitPage page;
	gen_mov_qword_to_reg_imm(HOST_R3, imm);
	return page.seal<uint64_t (*)()>()();
}

// Emit `gen_mov_dword_to_reg_imm(r3, imm)` and return the loaded value.
// On POWER10 a wide constant uses one pli instead of lis+ori.
uint32_t emit_mov_dword(uint32_t imm)
{
	JitPage page;
	gen_mov_dword_to_reg_imm(HOST_R3, imm);
	return page.seal<uint32_t (*)()>()();
}

// Emit `gen_add_imm(r3, imm)` and run it on `base` -> r3.
// On POWER10 a wide immediate uses one paddi instead of addis+addi.
uint32_t emit_add_imm(uint32_t base, uint32_t imm)
{
	JitPage page;
	gen_add_imm(HOST_R3, imm);
	return page.seal<uint32_t (*)(uint32_t)>()(base);
}

// Count host *instructions* emitted by `emit` (a prefixed instruction is one
// instruction occupying two words: its prefix word has primary opcode 1).
template <typename F>
size_t count_insns(F emit)
{
	JitPage page;
	uint32_t* p = reinterpret_cast<uint32_t*>(cache.pos);
	emit();
	uint32_t* const end = reinterpret_cast<uint32_t*>(cache.pos);
	size_t n = 0;
	while (p < end) {
		p += ((*p >> 26) == 1) ? 2 : 1; // prefix word -> 8-byte instruction
		++n;
	}
	return n;
}

// Emit one of the inlined operators from gen_fill_function_ptr() into a
// patch stanza and run it as f(op1_in_r3, op2_in_r4) -> r3.
uint32_t emit_op(unsigned flags_type, uint32_t op1, uint32_t op2)
{
	JitPage page;
	int dummy = 0;
	gen_fill_function_ptr(page.base(), &dummy, flags_type);
	// gen_fill_function_ptr writes at most 7 words (op[0..6]); terminate.
	reinterpret_cast<uint32_t*>(page.base())[7] = PpcBlr;
	cache.pos = page.base() + 8 * sizeof(uint32_t);
	__builtin___clear_cache(reinterpret_cast<char*>(page.base()),
	                        reinterpret_cast<char*>(cache.pos));
	EXPECT_EQ(mprotect(page.base(), 4096, PROT_READ | PROT_EXEC), 0);
	auto fn = reinterpret_cast<uint32_t (*)(uint32_t, uint32_t)>(page.base());
	return fn(op1, op2);
}

// Emit a 3-operand inlined op (FC_OP1=r3, FC_OP2=r4, FC_OP3=r5) -> r3.
uint32_t emit_op3(unsigned flags_type, uint32_t op1, uint32_t op2, uint32_t op3)
{
	JitPage page;
	int dummy = 0;
	gen_fill_function_ptr(page.base(), &dummy, flags_type);
	reinterpret_cast<uint32_t*>(page.base())[7] = PpcBlr;
	cache.pos = page.base() + 8 * sizeof(uint32_t);
	__builtin___clear_cache(reinterpret_cast<char*>(page.base()),
	                        reinterpret_cast<char*>(cache.pos));
	EXPECT_EQ(mprotect(page.base(), 4096, PROT_READ | PROT_EXEC), 0);
	auto fn = reinterpret_cast<uint32_t (*)(uint32_t, uint32_t, uint32_t)>(page.base());
	return fn(op1, op2, op3);
}

constexpr uint32_t kValues[] = {
	0x00000000, 0xFFFFFFFF, 0xDEADBEEF, 0x12345678,
	0xA5A5A5A5, 0x80000000, 0x00000001, 0x7FFFFFFF,
};

uint32_t rotl32(uint32_t x, unsigned n) { return n ? (x << n) | (x >> (32 - n)) : x; }
uint32_t rotr32(uint32_t x, unsigned n) { return n ? (x >> n) | (x << (32 - n)) : x; }
uint32_t rotl8(uint8_t v, unsigned n)   { return ((v << n) | (v >> (8 - n))) & 0xFF; }
uint32_t rotr8(uint8_t v, unsigned n)   { return ((v >> n) | (v << (8 - n))) & 0xFF; }
uint32_t rotl16(uint16_t v, unsigned n) { return ((v << n) | (v >> (16 - n))) & 0xFFFF; }
uint32_t rotr16(uint16_t v, unsigned n) { return ((v >> n) | (v << (16 - n))) & 0xFFFF; }

// --- gen_and_imm ------------------------------------------------------------
// Regression coverage for the high-halfword mask: the two-instruction fallback
// (mask spanning both halfwords, not a single contiguous run) must AND with
// imm>>16, not the boolean imm>16.
TEST(DynrecPpc64le, AndImmExhaustivePatterns)
{
	constexpr uint32_t masks[] = {
		// split path (set bits in both halfwords, non-contiguous):
		0xFF00FF00, 0x00FF00FF, 0xF0F0F0F0, 0x0F0F0F0F,
		0x80008000, 0xC003C003, 0xAAAA5555, 0x13572468,
		// other paths (single run / single halfword / all-ones):
		0xFFFFFFFF, 0x00000000, 0x0000FFFF, 0xFFFF0000,
		0x00FFFF00, 0x000000FF, 0xFF000000, 0x0FFFFFF0,
		0xFFFF00FF, 0xFF00FFFF,
	};
	for (uint32_t mask : masks)
		for (uint32_t value : kValues)
			EXPECT_EQ(emit_and_imm(value, mask), value & mask)
			        << "mask=" << std::hex << mask << " value=" << value;
}

// --- 64-bit constant / address materialization -----------------------------
// Covers the ISA 3.1 pli path (|imm| within signed 34 bits), the multi-
// instruction fallback (larger constants), and sign-extension edge cases.
TEST(DynrecPpc64le, MovQwordToRegImm)
{
	const uint64_t vals[] = {
		0x0, 0x1, 0xFFFF, 0x8000, 0x10000,
		0x80000000ULL, 0xFFFFFFFFULL,        // 32-bit
		0x100000000ULL, 0x1FFFFFFFFULL,      // top of the signed-34 pli range
		0x200000000ULL,                      // just past it -> fallback
		0xFFFFFFFFFFFFFFFFULL,               // -1, sign-extended by pli
		0x7FFFFFFFFFFFFFFFULL, 0x8000000000000000ULL,
		0xDEADBEEFCAFE1234ULL, 0x123456789ABCDEF0ULL,
	};
	for (uint64_t v : vals)
		EXPECT_EQ(emit_mov_qword(v), v) << "imm=" << std::hex << v;
}

// Reports (and bounds) the host code size for materializing a code-cache-local
// address: 1 prefixed instruction under ISA 3.1, otherwise the lis/ori chain.
TEST(DynrecPpc64le, MaterializationCodeSize)
{
	JitPage page;
	uint8_t* const start = cache.pos;
	// A target inside the page is always within PC-relative (pla) range.
	gen_mov_qword_to_reg_imm(HOST_R3,
	                         reinterpret_cast<uint64_t>(page.base()) + 64);
	const size_t bytes = static_cast<size_t>(cache.pos - start);
	std::printf("[   INFO   ] gen_mov_qword(near addr): drc_isa31=%d -> %zu bytes\n",
	            static_cast<int>(drc_isa31), bytes);
	if (drc_isa31)
		EXPECT_LE(bytes, 12u); // 8-byte pla (+ at most one alignment nop)
	else
		EXPECT_LE(bytes, 20u); // up to 5 x 4-byte instructions
}

// 32-bit constant load: pli fast path (POWER10) vs lis/ori, plus the small-
// immediate li path. Only the low 32 bits are defined for this helper.
TEST(DynrecPpc64le, MovDwordToRegImm)
{
	const uint32_t vals[] = {
		0x0, 0x1, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x7FFF0000,
		0x80000000, 0xFFFF0000, 0x12345678, 0xDEADBEEF, 0xFFFFFFFF,
	};
	for (uint32_t v : vals)
		EXPECT_EQ(emit_mov_dword(v), v) << "imm=" << std::hex << v;
}

// 32-bit immediate add: paddi fast path (POWER10) vs addis/addi, plus the
// no-op (0), single-addi and single-addis paths. Compared at 32 bits.
TEST(DynrecPpc64le, AddImm)
{
	const uint32_t bases[] = {0x0, 0x1000, 0xFFFFFFFF, 0x7FFFFFFF, 0xABCDEF01};
	const uint32_t imms[]  = {0x0, 0x1, 0x7FFF, 0x8000, 0x10000, 0x12345678,
	                          0x80000000, 0xFFFF8000, 0xFFFFFFFF};
	for (uint32_t base : bases)
		for (uint32_t imm : imms)
			EXPECT_EQ(emit_add_imm(base, imm), base + imm)
			        << "base=" << std::hex << base << " imm=" << imm;
}

// Demonstrates the instruction-count reduction: a wide immediate that needs
// two instructions on the portable path collapses to one prefixed instruction
// under ISA 3.1.
TEST(DynrecPpc64le, WideImmInstructionCount)
{
	const uint32_t wide = 0x12345678; // needs both halves
	const size_t add_n = count_insns([&] { gen_add_imm(HOST_R3, wide); });
	const size_t mov_n = count_insns([&] { gen_mov_dword_to_reg_imm(HOST_R3, wide); });
	std::printf("[   INFO   ] drc_isa31=%d: gen_add_imm=%zu insn, "
	            "gen_mov_dword=%zu insn (wide imm)\n",
	            static_cast<int>(drc_isa31), add_n, mov_n);
	if (drc_isa31) {
		EXPECT_EQ(add_n, 1u);
		EXPECT_EQ(mov_n, 1u);
	} else {
		EXPECT_EQ(add_n, 2u);
		EXPECT_EQ(mov_n, 2u);
	}
}

// --- inlined logical/arithmetic operators -----------------------------------
TEST(DynrecPpc64le, InlinedBinaryOperators)
{
	for (uint32_t a : kValues) {
		for (uint32_t b : kValues) {
			EXPECT_EQ(emit_op(t_ADDd, a, b), a + b);
			EXPECT_EQ(emit_op(t_ORd,  a, b), a | b);
			EXPECT_EQ(emit_op(t_ANDd, a, b), a & b);
			EXPECT_EQ(emit_op(t_SUBd, a, b), a - b);
			EXPECT_EQ(emit_op(t_XORd, a, b), a ^ b);
		}
	}
}

TEST(DynrecPpc64le, InlinedUnaryOperators)
{
	for (uint32_t a : kValues) {
		EXPECT_EQ(emit_op(t_INCd, a, 0), a + 1);
		EXPECT_EQ(emit_op(t_DECd, a, 0), a - 1);
		EXPECT_EQ(emit_op(t_NEGd, a, 0), static_cast<uint32_t>(-a));
	}
}

// Byte/word shift variants. Operands are passed pre-zero-extended (as the
// decoder loads them via lbz/lhz), and results are compared at the operand
// width. SAR must sign-extend the operand before shifting.
TEST(DynrecPpc64le, InlinedByteWordShifts)
{
	const uint8_t bytes[] = {0x01, 0x7F, 0x80, 0xF0, 0xFF};
	for (uint8_t b : bytes) {
		for (unsigned n : {1u, 2u, 7u}) {
			const uint32_t in = b; // zero-extended byte, as loaded by lbz
			EXPECT_EQ(emit_op(t_SHLb, in, n) & 0xFF,
			          static_cast<uint32_t>((b << n) & 0xFF)) << "SHLb";
			EXPECT_EQ(emit_op(t_SHRb, in, n) & 0xFF,
			          static_cast<uint32_t>(b >> n)) << "SHRb";
			EXPECT_EQ(emit_op(t_SARb, in, n) & 0xFF,
			          static_cast<uint32_t>(static_cast<int8_t>(b) >> n) & 0xFF)
			        << "SARb b=" << std::hex << (int)b << " n=" << n;
		}
	}
	const uint16_t words[] = {0x0001, 0x7FFF, 0x8000, 0xF000, 0xFFFF};
	for (uint16_t w : words) {
		for (unsigned n : {1u, 8u, 15u}) {
			const uint32_t in = w; // zero-extended word, as loaded by lhz
			EXPECT_EQ(emit_op(t_SHRw, in, n) & 0xFFFF,
			          static_cast<uint32_t>(w >> n)) << "SHRw";
			EXPECT_EQ(emit_op(t_SARw, in, n) & 0xFFFF,
			          static_cast<uint32_t>(static_cast<int16_t>(w) >> n) & 0xFFFF)
			        << "SARw w=" << std::hex << w << " n=" << n;
		}
	}
}

// Shift/rotate counts are tested in the range [0,31] where PPC slw/srw/sraw and
// the rlwnm-based rotates agree with x86's 32-bit semantics.
TEST(DynrecPpc64le, InlinedShiftsAndRotates)
{
	for (uint32_t a : kValues) {
		for (unsigned n : {0u, 1u, 7u, 8u, 15u, 16u, 17u, 31u}) {
			EXPECT_EQ(emit_op(t_SHLd, a, n), a << n) << "n=" << n;
			EXPECT_EQ(emit_op(t_SHRd, a, n), a >> n) << "n=" << n;
			EXPECT_EQ(emit_op(t_SARd, a, n),
			          static_cast<uint32_t>(static_cast<int32_t>(a) >> n))
			        << "n=" << n;
			EXPECT_EQ(emit_op(t_ROLd, a, n), rotl32(a, n)) << "n=" << n;
			EXPECT_EQ(emit_op(t_RORd, a, n), rotr32(a, n)) << "n=" << n;
		}
	}
}

// Byte/word rotates use an rlwimi to replicate the operand before rotlw.
TEST(DynrecPpc64le, InlinedRotatesByteWord)
{
	const uint8_t bytes[] = {0x01, 0x80, 0xF0, 0xA5, 0xFF};
	for (uint8_t b : bytes) {
		for (unsigned n : {1u, 3u, 7u}) {
			EXPECT_EQ(emit_op(t_ROLb, b, n) & 0xFF, rotl8(b, n))
			        << "ROLb b=" << std::hex << (int)b << " n=" << n;
			EXPECT_EQ(emit_op(t_RORb, b, n) & 0xFF, rotr8(b, n))
			        << "RORb b=" << std::hex << (int)b << " n=" << n;
		}
	}
	const uint16_t words[] = {0x0001, 0x8000, 0xF00F, 0xA5A5, 0xFFFF};
	for (uint16_t w : words) {
		for (unsigned n : {1u, 8u, 15u}) {
			EXPECT_EQ(emit_op(t_ROLw, w, n) & 0xFFFF, rotl16(w, n))
			        << "ROLw w=" << std::hex << w << " n=" << n;
			EXPECT_EQ(emit_op(t_RORw, w, n) & 0xFFFF, rotr16(w, n))
			        << "RORw w=" << std::hex << w << " n=" << n;
		}
	}
}

// SHLD/SHRD (double-precision shifts), 32-bit form, counts in [1,31].
TEST(DynrecPpc64le, InlinedDoubleShifts)
{
	for (uint32_t a : kValues) {
		for (uint32_t b : kValues) {
			for (unsigned n : {1u, 7u, 16u, 31u}) {
				EXPECT_EQ(emit_op3(t_DSHLd, a, b, n),
				          static_cast<uint32_t>((a << n) | (b >> (32 - n))))
				        << "DSHLd n=" << n;
				EXPECT_EQ(emit_op3(t_DSHRd, a, b, n),
				          static_cast<uint32_t>((a >> n) | (b << (32 - n))))
				        << "DSHRd n=" << n;
			}
		}
	}
}

} // namespace

#endif // ppc64le
