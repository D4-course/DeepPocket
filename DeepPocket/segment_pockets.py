'''
Segment out pocket shapes from top ranked pockets
'''
# import sys
# import os
# import logging
import argparse
# import wandb
import torch
from torch import nn
import numpy as np

import molgrid
# from skimage.morphology import binary_dilation
# from skimage.morphology import cube
# pylint: disable=E1101,R0913,R0914
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.measure import label
from scipy.spatial.distance import cdist
from prody import writePDB, parsePDB
from unet import Unet

def preprocess_output(inp, threshold):
    '''
    Preprocess output from UNET using threshold, dilation, and clearing border
    '''
    inp[inp>=threshold]=1
    inp[inp!=1]=0
    inp=inp.numpy()
    bw_ = closing(inp).any(axis=0)
    # remove artifacts connected to border
    cleared = clear_border(bw_)

    # label regions
    label_image, num_labels = label(cleared, return_num=True)
    largest=0
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum()
        if pocket_size >largest:
            largest=pocket_size
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum()
        if pocket_size <largest:
            label_image[np.where(pocket_idx)] = 0
    label_image[label_image>0]=1
    return torch.tensor(label_image,dtype=torch.float32)

def get_model_gmaker_eproviders(args):
    '''
    Prepare gridmaker and dataproviders
    '''
    # test example provider
    eptest_l = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False,
                        iteration_scheme=molgrid.IterationScheme.LargeEpoch,default_batch_size=1)
    eptest_l.populate(args.test_types)
    # gridmaker with defaults
    gmaker_img_l = molgrid.GridMaker(dimension=32)

    return  gmaker_img_l, eptest_l

def output_coordinates(tensor,center,dimension=16.25,resolution=0.5):
    '''
    get coordinates of mask from predicted mask
    '''
    tensor=tensor.numpy()
    indices = np.argwhere(tensor>0).astype('float32')
    indices *= resolution
    center=np.array([float(center[0]), float(center[1]), float(center[2])])
    indices += center
    indices -= dimension
    return indices

def predicted_aa(indices,prot_prody,distance):
    '''
    amino acids from mask distance thresholds
    '''
    prot_coords = prot_prody.getCoords()
    ligand_dist = cdist(indices, prot_coords)
    binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
    #get predicted protein residue indices involved in binding site
    prot_resin = prot_prody.getResindices()
    prot_binding_indices = prot_resin[binding_indices]
    prot_binding_indices = sorted(list(set(prot_binding_indices)))
    return prot_binding_indices

def output_pocket_pdb(pocket_name,prot_prody,pred_aa):
    '''
    output pocket pdb
    skip if no amino acids predicted
    '''
    if len(pred_aa)==0:
        return
    sel_str= 'resindex '
    for i in pred_aa:
        sel_str+= str(i)+' or resindex '
    sel_str=' '.join(sel_str.split()[:-2])
    pocket=prot_prody.select(sel_str)
    writePDB(pocket_name,pocket)

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('--test_types', type=str, required=True,
                        help="test types file")
    parser.add_argument('--model_weights', type=str, required=True,
                        help="weights for UNET")
    parser.add_argument('-t', '--threshold', type=float, required=False,
                        help="threshold for segmentation", default=0.5)
    parser.add_argument('-r', '--rank', type=int, required=False,
                        help="number of pockets to segment", default=1)
    parser.add_argument('--upsample', type=str, required=False,
                        help="Type of Upsampling", default=None)
    parser.add_argument('--num_classes', type=int, required=False,
                        help="Output channels for predicted masks, default 1", default=1)
    parser.add_argument('--dx_name', type=str, required=True,
                        help="dx file name")
    parser.add_argument('-p','--protein', type=str, required=False,
                         help="pdb file for predicting binding sites")
    parser.add_argument('--mask_dist', type=float, required=False,
                        help="distance from mask to residues", default=3.5)
    args_l = parser.parse_args(argv)

    argdict = vars(args_l)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += f' --{name}={val}'

    return (args_l, line)

def test(model, test_loader, gmaker_img,device,dx_name, args):
    '''
    test model on test_loader and save predicted masks
    '''
    if args.rank==0:
        return
    count=0
    model.eval()
    dims = gmaker_img.grid_dimensions(test_loader.num_types())
    tensor_shape = (1,) + dims
    #create tensor for input, centers and indices
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
    float_labels = torch.zeros((1, 4), dtype=torch.float32, device=device)
    prot_prody=parsePDB(args.protein)
    for batch in test_loader:
        count+=1
        # update float_labels with center and index values
        batch.extract_labels(float_labels)
        centers = float_labels[:, 1:]
        for b_ind in range(1):
            center = molgrid.float3(float(centers[b_ind][0]),
                                     float(centers[b_ind][1]), float(centers[b_ind][2]))
            # Update input tensor with b_ind'th datapoint of the batch
            gmaker_img.forward(center, batch[b_ind].coord_sets[0], input_tensor[b_ind])
        # Take only the first 14 channels as that is for proteins,
        #  other 14 are ligands and will remain 0.
        masks_pred = model(input_tensor[:, :14])
        masks_pred=masks_pred.detach().cpu()
        masks_pred=preprocess_output(masks_pred[0], args.threshold)
        # predict binding site residues
        pred_coords = output_coordinates(masks_pred, center)
        pred_aa = predicted_aa(pred_coords, prot_prody, args.mask_dist)
        output_pocket_pdb(dx_name+'_pocket'+str(count)+'.pdb',prot_prody,pred_aa)
        masks_pred=masks_pred.cpu()
        # Output predicted mask in .dx format
        masks_pred=molgrid.Grid3f(masks_pred)
        molgrid.write_dx(dx_name+'_'+str(count)+'.dx',masks_pred,center,0.5,1.0)
        if count>=args.rank:
            break

if __name__ == "__main__":
    (args, cmdline) = parse_args()
    gmaker_img, eptest = get_model_gmaker_eproviders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(args.num_classes, args.upsample)
    model.to(device)
    checkpoint = torch.load(args.model_weights)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    dx_name=args.dx_name
    test(model, eptest, gmaker_img,device,dx_name, args)
