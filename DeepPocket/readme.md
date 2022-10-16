# Visualising the DeepPocket outputs

This folder contains the code for visualising the outputs of the DeepPocket algorithm. For visualtion, Install pymol tool. For installation refer to the following url. http://pymol.org/2/.

```
    .dx files are the volumetric data files. These files are used to visualise the segmented pockets using DeepPocket.
    protein_nowat_pocketX.pdb files are the residue files. These files are used to visualise the residues.
```

## Visualising Input protein

Open pymol and load the input protein. For example, if the input protein is 1a0j.pdb, then type the following command in pymol.

```
load 1a0j.pdb
```

or you can use the menu bar to load the protein.

## Visualising the DeepPocket Output Residue

Open pymol and load the input protein. For example, if the input protein is 1a0j.pdb, also load any residue pdb file. For example, if the residue file is protein_nowat_pocket1.pdb. Use the rightside menu bar to visualise the residue file.

## Visualising the DeepPocket segmented pockets

>Open pymol and load the input protein. For example, if the input protein is 1a0j.pdb, also load any .dx file. 

>Once the .dx file is loaded, go to 'setting' menu and under 'surface', select 'cavity and pocket only'. 

>Then, in the  right side bar, you have the dx file name with options 'A, S, H, L, C'. Select S (means show), and choose 'everything' or 'dots'. 

>Then you can visualise the grid points of the segment pocket. Now use the 'A' (means Action) and choose the kind of surface visualisation you want.  