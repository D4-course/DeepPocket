'''
Takes a PDB file and removes hetero atoms from its structure.
First argument is path to original file, second argument is path to generated file
'''
import sys
from Bio.PDB import PDBParser, PDBIO, Select
import Bio

class NonHetSelect(Select):
    '''
    Classifies the residue as a amino acid or not
    '''
    def accept_residue(self, residue):
        return 1 if Bio.PDB.Polypeptide.is_aa(residue,standard=True) else 0

def clean_pdb(input_file,output_file):
    '''
    Takes a PDB file and removes hetero atoms from its structure.
    First argument is path to original file, second argument is path to generated file
    '''
    pdb = PDBParser().get_structure("protein", input_file)
    io = PDBIO() # pylint: disable=C0103
    io.set_structure(pdb)
    io.save(output_file, NonHetSelect())

if __name__ == '__main__':
    clean_pdb(sys.argv[1],sys.argv[2])
