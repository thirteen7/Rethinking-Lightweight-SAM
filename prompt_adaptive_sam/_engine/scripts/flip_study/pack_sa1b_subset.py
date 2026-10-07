# Extracted from the verified research implementation; see LICENSE and NOTICE.
import hashlib

def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()
