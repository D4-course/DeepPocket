# DeepPocket: Ligand Binding Site Detection and Segmentation using 3D Convolutional Neural Networks

## Requirements: 
- Docker
- CUDA >11.6

## Installation Instructions:
- Open a terminal and clone the repo
```
git clone https://github.com/D4-course/DeepPocket.git
```
- Download [trained models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/rishal_aggarwal_research_iiit_ac_in/EoJSrvuiKPlAluOJLjTzfpcBT2fVRdq8Sr4BMmil0_tvHw?e=Kj7reS) and load them into ```DeepPocket``` folder in the parent repo. You can download any classified_models and segmentation_models for this purpose or train one yourself and use. (Refer to [manual](./MANUAL.md))
- Execute ```run.sh``` that will build the docker image, run a container where the frontend and backend will be executed automatically
```
./run.sh
or
sh run.sh
```
- Navigate to the network or external URL that will be displayed on the terminal
```

You can now view your Streamlit app in your browser.

  Network URL: http://10.42.0.88:8501
  External URL: http://10.1.34.46:8501
```
## Website Instructions
- Firstly, select a protein file (```.pdb```) of interest and upload it. (Sample ```protein.pdb``` file attached in the repo)
- If you would like to view the segmented pockets, tick the ```segment the centers?``` option
- Click on ```predict``` to get a list of top pockets for the provided protein. This might upto 1min depending your computer architecture.
- An interactive structure of the input protein can be viewed along with the list of centers. 
- Segmented pockets (if chosen) can be downloaded as a ```.zip``` file by clicking on the download button. Please see the readme in zip folder for visualisation instructions. 
# Visualising the DeepPocket outputs

This folder contains the code for visualising the outputs of the DeepPocket algorithm. For visualtion, Install pymol tool. For installation refer to the following url. http://pymol.org/2/.

```
    .dx files are the segmented pockets. These files are used to visualise the segmented pockets using DeepPocket.
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

(For Refernce only)![segmented_pocket](https://user-images.githubusercontent.com/57574795/196051563-ad175fe5-7052-4eaf-9bd3-9f7d01304432.png)
