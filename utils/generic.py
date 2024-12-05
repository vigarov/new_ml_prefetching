from itertools import islice

def chunk_data(data, chunk_size):
    iterator = iter(data)
    return iter(lambda: list(islice(iterator, chunk_size)), [])
