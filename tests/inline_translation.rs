//! Differential test of the JIT's inline address translation against the interpreter.
//!
//! Every access lands on or next to a region boundary, a stack frame gap, a read-only region
//! or an unmapped region index, in the aligned memory mapping used by Agave.
#![cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]

use solana_sbpf::{
    assembler::assemble,
    ebpf,
    memory_region::MemoryRegion,
    program::{BuiltinProgram, SBPFVersion},
    test_utils::{run_program_with_input, ExecutionOutcome, TestContextObject},
    vm::{Config, ExecutionMode},
};
use std::sync::Arc;

const INPUT_LEN: usize = 32;

fn run_both(source: &str, sbpf_version: SBPFVersion) -> (String, String) {
    let config = Config {
        aligned_memory_mapping: true,
        enabled_sbpf_versions: sbpf_version..=sbpf_version,
        ..Config::default()
    };
    let loader = Arc::new(BuiltinProgram::new_loader(config));
    let executable = assemble::<TestContextObject>(source, loader).unwrap();
    let describe = |outcome: &ExecutionOutcome, mem: &[u8; INPUT_LEN]| {
        format!(
            "{:?} count={} pc={} mem={:?}",
            outcome.result, outcome.instruction_count, outcome.final_pc, mem
        )
    };
    let mut mem_interpreter = [0xAAu8; INPUT_LEN];
    let interpreter = run_program_with_input(
        &executable,
        &mut TestContextObject::new(100),
        MemoryRegion::new(&raw mut mem_interpreter, ebpf::MM_INPUT_START),
        &mut ExecutionMode::Interpreted,
    );
    executable.jit_compile().unwrap();
    let mut mem_jit = [0xAAu8; INPUT_LEN];
    let jit = run_program_with_input(
        &executable,
        &mut TestContextObject::new(100),
        MemoryRegion::new(&raw mut mem_jit, ebpf::MM_INPUT_START),
        &mut ExecutionMode::Jit,
    );
    (
        describe(&interpreter, &mem_interpreter),
        describe(&jit, &mem_jit),
    )
}

fn assert_same(source: &str) {
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V3] {
        let (interpreter, jit) = run_both(source, sbpf_version);
        assert_eq!(interpreter, jit, "{sbpf_version:?}: {source}");
    }
}

const OPS: [(&str, i64); 12] = [
    ("ldxb r0, [r1{off}]", 1),
    ("ldxh r0, [r1{off}]", 2),
    ("ldxw r0, [r1{off}]", 4),
    ("ldxdw r0, [r1{off}]", 8),
    ("stb [r1{off}], 0x11", 1),
    ("sth [r1{off}], 0x1122", 2),
    ("stw [r1{off}], 0x11223344", 4),
    ("stdw [r1{off}], 0x11223344", 8),
    ("stxb [r1{off}], r2", 1),
    ("stxh [r1{off}], r2", 2),
    ("stxw [r1{off}], r2", 4),
    ("stxdw [r1{off}], r2", 8),
];

fn program(r1_setup: &str, op: &str, offset: i64) -> String {
    let op = op.replace("{off}", &format!("{offset:+}"));
    format!(
        "
        {r1_setup}
        mov64 r2, 0x11223344
        lsh64 r2, 32
        or64 r2, 0x55667788
        {op}
        exit"
    )
}

fn region(base_region: i64, base_offset: i64) -> String {
    format!("mov64 r1, {base_region}\n lsh64 r1, 32\n add64 r1, {base_offset}")
}

#[test]
fn test_input_region_boundaries() {
    let len = INPUT_LEN as i64;
    for (op, size) in OPS {
        for offset in [
            0,
            1,
            len - size,
            len - size + 1,
            len - 1,
            len,
            -1,
            -8,
            4096,
            -4096,
        ] {
            assert_same(&program(&region(4, 0), op, offset));
        }
        // The same accesses with the offset folded into the base register
        for offset in [0, len - size, len - size + 1, -1] {
            assert_same(&program(&region(4, offset), op, 0));
        }
    }
}

#[test]
fn test_unmapped_and_readonly_regions() {
    for (op, _size) in OPS {
        // Index 0: the program's read-only region, index 1: empty bytecode slot,
        // index 3: the heap, index 5 and beyond: no region at all
        for base_region in [0, 1, 3, 5, 6, 255, 0x7FFF_FFFF, -1] {
            assert_same(&program(&region(base_region, 0), op, 0));
            assert_same(&program(&region(base_region, 0), op, -1));
        }
    }
}

#[test]
fn test_stack_frame_gaps() {
    let frame = 4096i64;
    for (op, size) in OPS {
        for offset in [
            -size,
            -size - 1,
            -frame,
            -frame - 1,
            -frame + 1,
            0,
            1,
            -1,
            frame - size,
            frame,
            frame + 1,
            2 * frame - size,
            2 * frame,
            2 * frame + frame / 2,
            3 * frame,
            64 * 2 * frame,
            -(64 * 2 * frame),
        ] {
            // Large offsets do not fit the 16 bit offset field, so they go through the base register
            assert_same(&program(
                &format!("mov64 r1, r10\n add64 r1, {offset}"),
                op,
                0,
            ));
        }
    }
}

#[test]
fn test_stack_gaps_inside_calls() {
    // The callee frame sits one frame and one gap above the caller in V0
    for offset in [-8, 0, 8, 4088, 4096, 8184, 8192, -4096, -4104] {
        let off = format!("{offset:+}");
        let source = format!(
            "
            stdw [r10-8], 7
            call function_foo
            exit
            function_foo:
            ldxdw r0, [r10{off}]
            stxdw [r10{off}], r0
            exit"
        );
        assert_same(&source);
    }
}
