FLTRACE_OUTPUT_PREFIX = "fltrace-data-"
FLTRACE_OUTPUT_SUFFIX = ".out"

TEXT_SEPARATOR = '_'*30

PAGE_SIZE = 4 * 1024

def get_page_address(address:int|str) -> int:
    if isinstance(address,str):
        address = int(address,16)
    return address & ~(PAGE_SIZE-1)

def get_page_num(address:int|str) -> int:
    return get_page_address(address) >> (PAGE_SIZE-1).bit_length()