'''
Takes the *_out/pockets directory from fpocket as input and outputs a file containining
 candidate pocket centers in that directory
'''
import sys
import re
import os
import numpy as np


def get_centers(directory):
    '''
    Takes the *_out/pockets directory from fpocket as input and
    outputs a file containining candidate pocket centers in that directory
    '''
    # pylint: disable=R1732,W1514,W1401
    try:
        bary = open(directory+'/bary_centers.txt','w')
        for file in os.listdir(directory):
            centers = []
            masses = []
            if file.endswith('vert.pqr'):
                num = int(re.search(r'\d+', file).group())
                file_p = open(directory+'/'+file)
                for line in file_p:
                    if line.startswith('ATOM'):
                        center=list(map(float,
                        re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                                    ' '.join(line.split()[5:]))))[:3]
                        mass=float(line.split()[-1])
                        centers.append(center)
                        masses.append(mass)
                centers=np.asarray(centers)
                masses=np.asarray(masses)
                xyzm = (centers.T * masses).T
                xyzm_sum = xyzm.sum(axis=0) # find the total xyz*m for each element
                cog = xyzm_sum / masses.sum()
                bary.write(str(num) + '\t' + str(cog[0])+'\t'
                     + str(cog[1]) + '\t' + str(cog[2]) + '\n')
    except FileNotFoundError:
        assert False, f'No such directory: {directory}'
    except Exception as eve:
        assert False, f'Error: {eve}'

if __name__ == '__main__':
    get_centers(sys.argv[1])
