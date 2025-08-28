import os, struct, random, json
import tables
import pandas as pd

# ------------ CONFIG ------------
INCLUDE_START = 0x4000017c
INCLUDE_END   = 0x40002cd8

ELF_PATH      = "src/kleidiai_test.elf"  # your ELF binary
SAMPLE_COUNT  = 7000       # None = all in-window execs
RNG_SEED      = None    # None for non-deterministic
WORD_SIZE     = 4

FAULT_TYPE    = "data"
FAULT_MODEL   = "toggle"    # ARCHIE's toggle bitflip model
MAX_INSN_CNT  = 10000

MEMDUMP_ADDR  = 0x401faad8
MEMDUMP_LEN   = 24

HDF5_PATH     = "output/output_goldenrun.hdf5"
OUT_PATH      = "configs/instruction_fmla_permanent.json"
# --------------------------------

# Allowed bit positions (A64 FMLA by-element .8h)
ALLOWED_BITS = list(range(0,5)) + list(range(5,10)) + [10,11] + list(range(16,21)) + [21]

# FMLA-by-element .8h mask/value
FMLA_BYELEM_8H_MASK = 0xFFC0F000
FMLA_BYELEM_8H_VAL  = 0x4F001000

def read_tbinfo_from_hdf5(hdf5_file_path):
    with tables.open_file(hdf5_file_path, mode='r') as f:
        tbinfo_table = f.root.Goldenrun.tbinfo
        rows = []
        for row in tbinfo_table.iterrows():
            rows.append({
                'identity': row['identity'],
                'size': row['size'],
                'ins_count': row['ins_count'],
                'num_exec': row['num_exec'],
                'assembler': row['assembler'].decode('utf-8') if isinstance(row['assembler'], bytes) else row['assembler']
            })
    return pd.DataFrame(rows)

# ---- ELF VA → file offset mapping ----
def get_text_segments(elf_path):
    segs = []
    with open(elf_path, "rb") as f:
        if f.read(4) != b'\x7fELF':
            raise ValueError("Not an ELF file")
        f.seek(0x10)
        ei_class = struct.unpack("B", f.read(1))[0]
        if ei_class != 2:
            raise ValueError("Expected 64-bit ELF")
        f.seek(0x20)
        phoff = struct.unpack("<Q", f.read(8))[0]
        f.seek(0x36)
        phentsize = struct.unpack("<H", f.read(2))[0]
        phnum = struct.unpack("<H", f.read(2))[0]
        for i in range(phnum):
            f.seek(phoff + i * phentsize)
            p_type = struct.unpack("<I", f.read(4))[0]
            f.seek(4, 1)  # p_flags
            p_offset = struct.unpack("<Q", f.read(8))[0]
            p_vaddr = struct.unpack("<Q", f.read(8))[0]
            f.seek(8, 1)  # p_paddr
            p_filesz = struct.unpack("<Q", f.read(8))[0]
            p_memsz = struct.unpack("<Q", f.read(8))[0]
            segs.append((p_vaddr, p_offset, p_filesz))
    return segs

SEGMENTS = get_text_segments(ELF_PATH)

def va_to_file_offset(va):
    for vaddr, off, size in SEGMENTS:
        if vaddr <= va < vaddr + size:
            return off + (va - vaddr)
    return None

def read_u32_le_at_va(va):
    file_off = va_to_file_offset(va)
    if file_off is None:
        return None
    with open(ELF_PATH, "rb") as f:
        f.seek(file_off)
        b = f.read(4)
        if len(b) != 4:
            return None
        return struct.unpack("<I", b)[0]

def is_fmla_by_element_8h(word):
    return (word & FMLA_BYELEM_8H_MASK) == FMLA_BYELEM_8H_VAL

# ---- fault generation ----
def generate_fault_config_from_tbinfo(hdf5_file_path, include_start, include_end, sample_count=None):
    if RNG_SEED is not None:
        random.seed(RNG_SEED)

    df = read_tbinfo_from_hdf5(hdf5_file_path)
    fault_lifespan = [0]
    execs = []
    considered, matched = 0, 0
    seen_addrs = set()

    for _, row in df.iterrows():
        base = row['identity']
        ins_count = row['ins_count']
        num_exec = row['num_exec']
        for off in range(ins_count):
            addr = base + off * WORD_SIZE
            if not (include_start <= addr <= include_end):
                continue
            considered += 1
            word = read_u32_le_at_va(addr)
            if word is None or not is_fmla_by_element_8h(word):
                continue
            matched += 1
            seen_addrs.add(addr)
            for k in range(1, num_exec + 1):
                execs.append({'instruction_addr': addr, 'exec_count': k})

    print(f"Window: [{hex(include_start)} .. {hex(include_end)}]")
    print(f"Considered: {considered}, Matched FMLA .8h: {len(seen_addrs)}")
    print(f"Total execs matched: {len(execs)}")

    if sample_count is not None and sample_count < len(execs):
        faults = _sample_faults(execs, sample_count)
    else:
        faults = _all_faults(execs)

    config = {
        "max_instruction_count": MAX_INSN_CNT,
        "start": {"address": 0x40007ee4, "counter": 1},
        "end": {"address": 0x40008740, "counter": 1},
        "faults": faults,
        "memorydump": [{"address": MEMDUMP_ADDR, "length": MEMDUMP_LEN}],
        "mem_info": True
    }
    return config

def _fault_entry(addr, exec_count, bit):
    return [{
        "fault_address": [addr],
        "fault_type": FAULT_TYPE,
        "fault_model": FAULT_MODEL,
        "num_bytes": WORD_SIZE,
        "fault_lifespan": [0],
        "fault_mask": [1 << bit],
        "trigger_address": [-1],
        "trigger_counter": [exec_count]
    }]

def _sample_faults(execs, n):
    faults, used = [], set()
    while len(faults) < n:
        e = random.choice(execs)
        key = (e['instruction_addr'], e['exec_count'])
        if key in used:
            continue
        used.add(key)
        bit = random.choice(ALLOWED_BITS)
        faults.append(_fault_entry(e['instruction_addr'], e['exec_count'], bit))
    return faults

def _all_faults(execs):
    faults = []
    for i, e in enumerate(execs):
        if i % 10000 == 0:
            print(f"Processing {i}/{len(execs)}")
        bit = random.choice(ALLOWED_BITS)
        faults.append(_fault_entry(e['instruction_addr'], e['exec_count'], bit))
    return faults

# ---- run ----
if __name__ == "__main__":
    config = generate_fault_config_from_tbinfo(HDF5_PATH, INCLUDE_START, INCLUDE_END, sample_count=SAMPLE_COUNT)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved to: {OUT_PATH}")
    print(f"Faults generated: {len(config['faults'])}")