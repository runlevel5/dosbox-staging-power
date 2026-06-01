// SPDX-FileCopyrightText:  2020-2026 The DOSBox Staging Team
// SPDX-License-Identifier: GPL-2.0-or-later

// Execution tests for the ARMv8 (AArch64) dynamic recompiler code generator
// (src/cpu/core_dynrec/risc_armv8le.h), a sibling of the PPC64LE tests.
//
// Each test makes the generator emit real AArch64 machine code into an
// executable page, runs it, and compares the result against a reference
// computed in C++. It therefore only does anything on an aarch64 host; on every
// other architecture the translation unit is empty.
//
// As with the PPC64LE tests, the real header is #included inside an anonymous
// namespace with minimal stand-ins for the context it expects (code cache,
// Segs[]/cpu_regs[], core_dynrec), so the shipped code is tested directly
// without pulling in the whole CPU core and without clashing with the real
// symbols that dosboxcommon also defines.

#if defined(__aarch64__)

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/mman.h>
#include <unistd.h>

namespace {

using Bits = long;
using Bitu = unsigned long;

struct CacheCursor {
	uint8_t* pos = nullptr;
};
CacheCursor cache;

inline void cache_addd(uint32_t v, const uint8_t* pos)
{
	std::memcpy(const_cast<uint8_t*>(pos), &v, sizeof(v));
}
inline void cache_addd(uint32_t v)
{
	cache_addd(v, cache.pos);
	cache.pos += sizeof(v);
}
inline void cache_addw(uint16_t v, const uint8_t* pos)
{
	std::memcpy(const_cast<uint8_t*>(pos), &v, sizeof(v));
}
inline void cache_addb(uint8_t v, const uint8_t* pos)
{
	std::memcpy(const_cast<uint8_t*>(pos), &v, sizeof(v));
}
inline void cache_addq(uint64_t v)
{
	std::memcpy(cache.pos, &v, sizeof(v));
	cache.pos += sizeof(v);
}

// The generator only takes the addresses of these; the tests never dereference
// the emitted accesses to them.
uint64_t Segs[256];
uint64_t cpu_regs[256];
struct CoreDynrec {
	Bitu callback;
	Bitu readdata;
} core_dynrec;

// x86 flag-changing instruction types (mirrors the anonymous enum in
// src/cpu/lazyflags.h; gen_fill_function_ptr switches on these values).
enum {
	t_UNKNOWN = 0,
	t_ADDb, t_ADDw, t_ADDd, t_ORb, t_ORw, t_ORd,
	t_ADCb, t_ADCw, t_ADCd, t_SBBb, t_SBBw, t_SBBd,
	t_ANDb, t_ANDw, t_ANDd, t_SUBb, t_SUBw, t_SUBd,
	t_XORb, t_XORw, t_XORd, t_CMPb, t_CMPw, t_CMPd,
	t_INCb, t_INCw, t_INCd, t_DECb, t_DECw, t_DECd,
	t_TESTb, t_TESTw, t_TESTd, t_SHLb, t_SHLw, t_SHLd,
	t_SHRb, t_SHRw, t_SHRd, t_SARb, t_SARw, t_SARd,
	t_ROLb, t_ROLw, t_ROLd, t_RORb, t_RORw, t_RORd,
	t_RCLb, t_RCLw, t_RCLd, t_RCRb, t_RCRw, t_RCRd,
	t_NEGb, t_NEGw, t_NEGd, t_DSHLw, t_DSHLd, t_DSHRw, t_DSHRd,
	t_MUL, t_DIV, t_NOTDONE, t_LASTFLAG
};

// Most of the header's static helpers are unused here; silence the warnings
// only for the include.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "cpu/core_dynrec/risc_armv8le.h"
#pragma GCC diagnostic pop

constexpr uint32_t ArmRet = 0xd65f03c0; // ret (x30)

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

	template <typename Fn>
	Fn seal()
	{
		__builtin___clear_cache(reinterpret_cast<char*>(mem_),
		                        reinterpret_cast<char*>(cache.pos));
		EXPECT_EQ(mprotect(mem_, kSize, PROT_READ | PROT_EXEC), 0);
		return reinterpret_cast<Fn>(mem_);
	}

private:
	static constexpr size_t kSize = 4096;
	uint8_t* mem_                  = nullptr;
};

uint32_t emit_and_imm(uint32_t value, uint32_t mask)
{
	JitPage page;
	gen_and_imm(HOST_r0, mask);
	cache_addd(ArmRet);
	return page.seal<uint32_t (*)(uint32_t)>()(value);
}
uint64_t emit_mov_qword(uint64_t imm)
{
	JitPage page;
	gen_mov_qword_to_reg_imm(HOST_r0, imm);
	cache_addd(ArmRet);
	return page.seal<uint64_t (*)()>()();
}
uint32_t emit_mov_dword(uint32_t imm)
{
	JitPage page;
	gen_mov_dword_to_reg_imm(HOST_r0, imm);
	cache_addd(ArmRet);
	return page.seal<uint32_t (*)()>()();
}
uint32_t emit_add_imm(uint32_t base, uint32_t imm)
{
	JitPage page;
	gen_add_imm(HOST_r0, imm);
	cache_addd(ArmRet);
	return page.seal<uint32_t (*)(uint32_t)>()(base);
}
// gen_fill_function_ptr writes a 5-word stanza (FC_OP1=w0, FC_OP2=w1,
// FC_OP3=w2, FC_RETOP=w0); append ret as the sixth word and run it.
uint32_t emit_op(unsigned flags_type, uint32_t a, uint32_t b, uint32_t c = 0)
{
	JitPage page;
	int dummy = 0;
	gen_fill_function_ptr(page.base(), &dummy, flags_type);
	reinterpret_cast<uint32_t*>(page.base())[5] = ArmRet;
	cache.pos = page.base() + 6 * sizeof(uint32_t);
	auto fn = page.seal<uint32_t (*)(uint32_t, uint32_t, uint32_t)>();
	return fn(a, b, c);
}

constexpr uint32_t kValues[] = {
	0x0, 0xFFFFFFFF, 0xDEADBEEF, 0x12345678,
	0xA5A5A5A5, 0x80000000, 0x1, 0x7FFFFFFF,
};

uint32_t rotl32(uint32_t x, unsigned n) { return n ? (x << n) | (x >> (32 - n)) : x; }
uint32_t rotr32(uint32_t x, unsigned n) { return n ? (x >> n) | (x << (32 - n)) : x; }
uint32_t rotl8(uint8_t v, unsigned n)   { return ((v << n) | (v >> (8 - n))) & 0xFF; }
uint32_t rotr8(uint8_t v, unsigned n)   { return ((v >> n) | (v << (8 - n))) & 0xFF; }
uint32_t rotl16(uint16_t v, unsigned n) { return ((v << n) | (v >> (16 - n))) & 0xFFFF; }
uint32_t rotr16(uint16_t v, unsigned n) { return ((v >> n) | (v << (16 - n))) & 0xFFFF; }

TEST(DynrecArmv8le, AndImmExhaustivePatterns)
{
	constexpr uint32_t masks[] = {
		0xFF00FF00, 0x00FF00FF, 0xF0F0F0F0, 0x0F0F0F0F,
		0x80008000, 0xC003C003, 0xAAAA5555, 0x13572468,
		0xFFFFFFFF, 0x00000000, 0x0000FFFF, 0xFFFF0000,
		0x00FFFF00, 0x000000FF, 0xFF000000, 0xFFFF00FF, 0xFF00FFFF,
	};
	for (uint32_t mask : masks)
		for (uint32_t value : kValues)
			EXPECT_EQ(emit_and_imm(value, mask), value & mask)
			        << "mask=" << std::hex << mask << " value=" << value;
}

TEST(DynrecArmv8le, MovQwordToRegImm)
{
	const uint64_t vals[] = {
		0x0, 0x1, 0xFFFF, 0x8000, 0x80000000ULL, 0xFFFFFFFFULL,
		0x100000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL,
		0xDEADBEEFCAFE1234ULL, 0x123456789ABCDEF0ULL,
	};
	for (uint64_t v : vals)
		EXPECT_EQ(emit_mov_qword(v), v) << "imm=" << std::hex << v;
}

TEST(DynrecArmv8le, MovDwordToRegImm)
{
	for (uint32_t v : kValues)
		EXPECT_EQ(emit_mov_dword(v), v) << "imm=" << std::hex << v;
}

TEST(DynrecArmv8le, AddImm)
{
	const uint32_t bases[] = {0x0, 0x1000, 0xFFFFFFFF, 0x7FFFFFFF};
	const uint32_t imms[]  = {0x0, 0x1, 0x7FFF, 0x8000, 0x10000,
	                          0x12345678, 0x80000000, 0xFFFFFFFF};
	for (uint32_t base : bases)
		for (uint32_t imm : imms)
			EXPECT_EQ(emit_add_imm(base, imm), base + imm)
			        << "base=" << std::hex << base << " imm=" << imm;
}

TEST(DynrecArmv8le, InlinedBinaryOperators)
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

TEST(DynrecArmv8le, InlinedUnaryOperators)
{
	for (uint32_t a : kValues) {
		EXPECT_EQ(emit_op(t_INCd, a, 0), a + 1);
		EXPECT_EQ(emit_op(t_DECd, a, 0), a - 1);
		EXPECT_EQ(emit_op(t_NEGd, a, 0), static_cast<uint32_t>(-a));
	}
}

TEST(DynrecArmv8le, InlinedByteWordShifts)
{
	for (uint8_t b : {0x01, 0x7F, 0x80, 0xF0, 0xFF}) {
		for (unsigned n : {1u, 2u, 7u}) {
			EXPECT_EQ(emit_op(t_SHLb, b, n) & 0xFF,
			          static_cast<uint32_t>((b << n) & 0xFF)) << "SHLb";
			EXPECT_EQ(emit_op(t_SHRb, b, n) & 0xFF,
			          static_cast<uint32_t>(b >> n)) << "SHRb";
			EXPECT_EQ(emit_op(t_SARb, b, n) & 0xFF,
			          static_cast<uint32_t>(static_cast<int8_t>(b) >> n) & 0xFF)
			        << "SARb b=" << std::hex << (int)b << " n=" << n;
		}
	}
	for (uint16_t w : {0x0001, 0x7FFF, 0x8000, 0xF000, 0xFFFF}) {
		for (unsigned n : {1u, 8u, 15u}) {
			EXPECT_EQ(emit_op(t_SHRw, w, n) & 0xFFFF,
			          static_cast<uint32_t>(w >> n)) << "SHRw";
			EXPECT_EQ(emit_op(t_SARw, w, n) & 0xFFFF,
			          static_cast<uint32_t>(static_cast<int16_t>(w) >> n) & 0xFFFF)
			        << "SARw w=" << std::hex << w << " n=" << n;
		}
	}
}

TEST(DynrecArmv8le, InlinedShiftsAndRotates)
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

TEST(DynrecArmv8le, InlinedRotatesByteWord)
{
	for (uint8_t b : {0x01, 0x80, 0xF0, 0xA5, 0xFF}) {
		for (unsigned n : {1u, 3u, 7u}) {
			EXPECT_EQ(emit_op(t_ROLb, b, n) & 0xFF, rotl8(b, n)) << "ROLb";
			EXPECT_EQ(emit_op(t_RORb, b, n) & 0xFF, rotr8(b, n)) << "RORb";
		}
	}
	for (uint16_t w : {0x0001, 0x8000, 0xF00F, 0xA5A5, 0xFFFF}) {
		for (unsigned n : {1u, 8u, 15u}) {
			EXPECT_EQ(emit_op(t_ROLw, w, n) & 0xFFFF, rotl16(w, n)) << "ROLw";
			EXPECT_EQ(emit_op(t_RORw, w, n) & 0xFFFF, rotr16(w, n)) << "RORw";
		}
	}
}

TEST(DynrecArmv8le, InlinedDoubleShifts)
{
	for (uint32_t a : kValues) {
		for (uint32_t b : kValues) {
			for (unsigned n : {1u, 7u, 16u, 31u}) {
				EXPECT_EQ(emit_op(t_DSHLd, a, b, n),
				          static_cast<uint32_t>((a << n) | (b >> (32 - n))))
				        << "DSHLd n=" << n;
				EXPECT_EQ(emit_op(t_DSHRd, a, b, n),
				          static_cast<uint32_t>((a >> n) | (b << (32 - n))))
				        << "DSHRd n=" << n;
			}
		}
	}
}

} // namespace

#endif // __aarch64__
