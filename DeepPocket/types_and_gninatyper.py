'''
creates types and gninatypes files of the protein for input to CNN via libmolgrid
first argument is path to protein file
second argument is path to barycenters list file
'''
import sys
import os
import pathlib
import struct
import molgrid
import numpy as np
# from model import Model
# pylint: disable=E1101,W1514,R1732,E1136

def gninatype(file):
    '''
    creates gninatype file for model input
    '''
    # creates gninatype file for model input
    f_p=open(file.replace('.pdb','.types'),'w')
    f_p.write(file)
    f_p.close()
    atom_map=molgrid.FileMappedGninaTyper(
                    f'{pathlib.Path(os.path.realpath(__file__)).resolve().parent}/gninamap')
    dataloader=molgrid.ExampleProvider(atom_map,shuffle=False,default_batch_size=1)
    train_types=file.replace('.pdb','.types')
    dataloader.populate(train_types)
    example=dataloader.next()
    coords=example.coord_sets[0].coords.tonumpy()
    types=example.coord_sets[0].type_index.tonumpy()
    types=np.int_(types)
    fout=open(file.replace('.pdb','.gninatypes'),'wb')
    for i in range(coords.shape[0]):
        fout.write(struct.pack('fffi',coords[i][0],coords[i][1],coords[i][2],types[i]))
    fout.close()
    os.remove(train_types)
    return file.replace('.pdb','.gninatypes')

def create_types(file,protein):
    '''
    create types file for model predictions
    '''
    # create types file for model predictions
    fout=open(file.replace('.txt','.types'),'w')
    fin =open(file,'r')
    for line in fin:
        fout.write(' '.join(line.split()) + ' ' + protein +'\n')
    return file.replace('.txt','.types')


if __name__ == '__main__':
    PROTEIN=gninatype(sys.argv[1])
    types_m=create_types(sys.argv[2],PROTEIN)
