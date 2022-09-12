'''Takes a bunch of types training files. First argument is what index the receptor starts on
Reads in the gninatypes files specified in these types files
 and writes out a monolithic receptor cache file.
Version 2 is optimized for memory mapped storage of caches.  keys (file names) are stored
first followed by dense storage of values (coordinates and types).

Thanks to David Koes for original script
(https://github.com/gnina/scripts/blob/master/create_caches2.py)
'''

import os
import sys
import struct
import argparse
# import traceback
import multiprocessing

mols_to_read = multiprocessing.Queue()
mols_to_write = multiprocessing.Queue()
N = multiprocessing.cpu_count() * 2

# pylint: disable=R1732,W1514, W0703
def read_data(data_root):
    '''read a types file and put it in mols_to_write'''
    while True:
        sys.stdout.flush()
        mol = mols_to_read.get()
        if mol is None:
            break
        file_name = mol
        if len(data_root):
            file_name = data_root + '/' + mol
        try:
            with open(file_name, 'rb') as gninatype:
                data = gninatype.read()
                assert len(data) % 16 == 0
                if len(data) == 0:
                    print(file_name, "EMPTY")
                else:
                    mols_to_write.put((mol, data))
        except Exception as excpt:
            print(file_name)
            print(excpt)
    mols_to_write.put(None)

def fill_queue(molfiles):
    'thread for filling mols_to_read'
    for mol in molfiles:
        mols_to_read.put(mol)
    for _ in range(N):
        mols_to_read.put(None)


def create_cache2(molfiles, data_root, outfile):
    '''Create an outfile molcache2 file from the list molfiles stored at data_root.'''
    # first byte is for versioning
    out = open(outfile, 'wb')
    out.write(struct.pack('i', -1))
    out.write(struct.pack('L', 0))  # placeholder for offset to keys

    filler = multiprocessing.Process(target=fill_queue, args=(molfiles,))
    filler.start()

    readers = multiprocessing.Pool(N)
    for _ in range(N):
        readers.apply_async(read_data, (data_root,))

    offsets = {}  # indxed by mol, location of data
    # start writing molecular data
    endcnt = 0
    while True:
        moldata = mols_to_write.get()
        if moldata is None:
            endcnt += 1
            if endcnt == N:
                break
            continue
        (mol, data) = moldata
        offsets[mol] = out.tell()
        natoms = len(data) // 16
        out.write(struct.pack('i', natoms))
        out.write(data)

    start = out.tell()  # where the names start
    for mol in molfiles:
        if len(mol) > 255:
            print("Skipping", mol, "since filename is too long")
            continue
        if mol not in offsets:
            print("SKIPPING", mol, "since failed to read it in")
            continue
        str1 = bytes(mol, encoding='UTF-8')
        out.write(struct.pack('B', len(str1)))
        out.write(str1)
        out.write(struct.pack('L', offsets[mol]))

    # now set start
    out.seek(4)
    out.write(struct.pack('L', start))
    out.seek(0, os.SEEK_END)
    out.close()


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--col', required=True, type=int, help='Column receptor starts on')
parser.add_argument('--recmolcache', default='rec.molcache2', type=str,
                     help='Filename of receptor cache')
parser.add_argument('-d', '--data_root', type=str, required=False,
                    help="Root folder for relative paths in train/test files", default='')
parser.add_argument('fnames', nargs='+', type=str, help='types files to process')

args = parser.parse_args()

# load all file names into memory
seenrec = set()
for fname in args.fnames:
    for line in open(fname):
        vals = line.split()
        rec = vals[args.col]

        if rec not in seenrec:
            seenrec.add(rec)

create_cache2(sorted(list(seenrec)), args.data_root, args.recmolcache)
