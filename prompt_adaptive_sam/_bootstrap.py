from pathlib import Path
import sys

ENGINE = Path(__file__).resolve().parent / '_engine'

def bootstrap():
    for path in (ENGINE, ENGINE/'.local-deps/TinySAM', ENGINE/'.local-deps/MobileSAM'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

bootstrap()
