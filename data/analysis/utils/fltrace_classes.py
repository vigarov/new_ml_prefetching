from dataclasses import dataclass, field
import re
from pathlib import Path
from bisect import bisect_left


# parse /proc/<pid>/maps
MAPS_LINE_RE = re.compile(r"""
    (?P<addr_start>[0-9a-f]+)-(?P<addr_end>[0-9a-f]+)\s+  # Address
    (?P<perms>\S+)\s+                                     # Permissions
    (?P<offset>[0-9a-f]+)\s+                              # Map offset
    (?P<dev>\S+)\s+                                       # Device node
    (?P<inode>\d+)\s+                                     # Inode
    (?P<path>.*)\s+                                   # path
""", re.VERBOSE)
# `Record` class was taken from fltrace's original parser
@dataclass
class Record:
    """A line in /proc/<pid>/maps"""
    addr_start: int
    addr_end: int
    perms: str
    offset: int
    path: str

    @staticmethod
    def parse(filename):
        records = []
        with open(filename) as fd:
            for line in fd:
                m = MAPS_LINE_RE.match(line)
                if not m:
                    print("Skipping: %s" % line)
                    continue
                addr_start, addr_end, perms, offset, _, _, path = m.groups()
                r = Record(addr_start=int(addr_start, 16), addr_end=int(addr_end, 16), offset=int(offset, 16),
                           perms=perms, path=path)
                records.append(r)
        return records

    @staticmethod
    def find_record(records, addr):
        for r in records:
            if r.addr_start <= addr < r.addr_end:
                return r
        return None



@dataclass
class ObjDump_ASM_Instr:
    addr: int
    hex_repr: str  # space separate
    instr: str
    params: str

    def get_full_text_repr(self):
        return self.instr + ' ' + self.params

    def __str__(self):
        return self.get_full_text_repr()


@dataclass
class ObjDump_Section:
    name: str
    start_addr: int  # included
    end_addr: int = -1  # excluded
    asm_instructions: list[ObjDump_ASM_Instr] = field(default_factory=list)


@dataclass
class ObjDump:
    sections: list[ObjDump_Section] = field(default_factory=list)  # sorted by section start address

# LibOrExe was adapted from fltrace's original parser
class LibOrExe:
    """A library or executable mapped into process memory"""
    records: list
    ips: list
    path: str
    base_addr: int
    codemap: dict
    objdump: ObjDump | None

    def __init__(self, records):
        """For libs collected from /proc/<pid>/maps"""
        self.records = records
        self.path = records[0].path
        self.base_addr = min([r.addr_start for r in records])
        self.ips = []
        self.codemap = {}
        self.objdump = None

def __get_od_file(objdump_dir,binary_file):
    od = Path(objdump_dir)
    assert od.exists() and od.is_dir()
    saught_file = Path(binary_file)
    od_file = None
    for file in od.glob('*'):
        if file.name == saught_file.name:
            od_file = file
            break
    return od_file

def get_objdump_object(objdump_dir:str,binary_file):
    od_file = __get_od_file(objdump_dir, binary_file)
    assert od_file is not None
    with open(od_file.absolute().as_posix(),'r',encoding="utf-8") as f:
        objdump_out = f.read()
    objdump = ObjDump()
    current_section = None
    new_big_section = False
    for line in objdump_out.split("\n")[3:]:
        if line is None:
            continue
        elif line.startswith("Disassembly of section"):
            new_big_section = True
        elif line.strip() != '':
            if line[0].isnumeric():
                # We're starting a new section
                assert ':' in line
                splitted = line.split()
                assert len(splitted) == 2
                raw_start, raw_name = splitted[0], splitted[1]
                assert raw_start.isalnum() and '<' in raw_name and '>' in raw_name
                curr_add = int(raw_start, 16)
                if current_section is not None:
                    # End previous section
                    assert new_big_section or current_section.end_addr == curr_add
                    objdump.sections.append(current_section)
                new_big_section = False
                # Start the new one
                current_section = ObjDump_Section(start_addr=curr_add,
                                                  name=raw_name)  # .replace('<','').replace('>','')) ?
            else:
                elements = line.strip().split('\t')
                assert ':' == elements[0][-1]
                curr_add = elements[0][:-1]
                assert curr_add.isalnum()
                curr_add = int(curr_add, 16)
                hex_repr = elements[1].strip()
                if len(elements) == 2:
                    # nop, quick path
                    instr = "nop"
                    params = ""
                else:
                    assert len(elements) == 3
                    textual_repr = elements[2] if len(elements) == 3 else "nop"
                    tr_splitted = textual_repr.split()
                    # for everything but `bnd <instr>`, instruction is one word, rest is params
                    # `bnd` simply specifies CPU to check bounds, can ignore it, as it doesn't give semantical info abt input
                    if "bnd" in textual_repr:
                        assert tr_splitted[0] == "bnd"
                        tr_splitted = tr_splitted[1:]
                    instr = tr_splitted[0]
                    # restore params with spaces (e.g.: for `call`)
                    params = ' '.join(tr_splitted[1:])
                curr_line_asm = ObjDump_ASM_Instr(curr_add, hex_repr, instr, params)
                current_section.asm_instructions.append(curr_line_asm)
        elif current_section is not None:
            # End Section
            last_asm_instr = current_section.asm_instructions[-1]
            current_section.end_addr = last_asm_instr.addr + len(last_asm_instr.hex_repr.split())
    if objdump.sections[-1] != current_section:
        objdump.sections.append(current_section)
    objdump.sections.sort(
        key=lambda section: section.start_addr)  # Should essentially not change the order, but juuuust in case
    return objdump


def binary_search(a, x, key, lo=0, hi=None):
    if hi is None: hi = len(a)
    pos = bisect_left(a, x, lo, hi, key=key)  # find insertion position
    return pos if pos != hi and key(a[pos]) == x else -1  # don't walk off the end


def get_surrounding_assembly(loe: LibOrExe, ip: int, window:tuple[int,int]) -> (list[ObjDump_ASM_Instr], str):
    correct_rec = Record.find_record(loe.records, ip)
    assert correct_rec
    ip = correct_rec.offset + (ip - correct_rec.addr_start)
    # returns element after which we can insert such that we remain sorted
    objdump = loe.objdump
    address_section_idx = bisect_left(objdump.sections, ip, lo=0, hi=len(objdump.sections),
                                      key=lambda section: section.start_addr)
    if address_section_idx == len(objdump.sections):
        address_section_idx-=1
        assert objdump.sections[address_section_idx].start_addr <= ip < objdump.sections[address_section_idx].end_addr
    elif objdump.sections[address_section_idx].start_addr != ip:
        assert address_section_idx != 0 and objdump.sections[address_section_idx].start_addr >= ip
        address_section_idx -= 1
    ip_sect = objdump.sections[address_section_idx]
    assert ip_sect.start_addr <= ip < ip_sect.end_addr
    asms_in_sect = ip_sect.asm_instructions
    ip_idx_in_list = binary_search(asms_in_sect, ip, lambda asm_inst: asm_inst.addr)
    assert ip_idx_in_list != -1
    past_len,future_len = window
    min_past, max_future = max(0, ip_idx_in_list - past_len), min(len(asms_in_sect), ip_idx_in_list + future_len)
    return asms_in_sect[min_past:max_future], ip_sect.name
