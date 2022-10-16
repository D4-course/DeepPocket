'''
Api for Deeppocket using Fastapi
    /api/v1/ - api version 1
    /api/v1/rank/protein/{protein}/num_pockets/{num_pockets} - rank pockets
    /api/v1/segment?protein=protein.pdb&num_pockets=10 - segment pockets
'''
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
import uvicorn
from predict import *
from zipfile import ZipFile
import io
from fastapi.responses import StreamingResponse
BASE_API = "/api/v1"


# function to predict ranks
app = FastAPI()

RANK_MODEL = "first_model_fold1_best_test_auc_85001.pth.tar"
SEG_MODEL = "seg0_best_test_IOU_91.pth.tar"
class Args:
    def __init__(self, class_checkpoint, protein, rank, seg_checkpoint):
        self.class_checkpoint = class_checkpoint
        self.seg_checkpoint = seg_checkpoint
        self.protein = protein
        self.rank = rank
        self.upsample = None
        self.num_classes = 1
        self.threshold = 0.5
        self.mask_dist = 3.5
def rank_pockets(protein, class_checkpoint, seg_checkpoint,num_pockets = 0 ):
    args = Args(class_checkpoint, protein, num_pockets, seg_checkpoint)
    #clean pdb file and remove hetero atoms/non standard residues
    protein_file=args.protein
    protein_nowat_file=protein_file.replace('.pdb','_nowat.pdb')
    clean_pdb(protein_file,protein_nowat_file)
    #fpocket
    os.system('fpocket -f '+protein_nowat_file)
    fpocket_dir=os.path.join(protein_nowat_file.replace('.pdb','_out'),'pockets')
    get_centers(fpocket_dir)
    barycenter_file=os.path.join(fpocket_dir,'bary_centers.txt')
    #types and gninatyper
    protein_gninatype=gninatype(protein_nowat_file)
    class_types=create_types(barycenter_file,protein_gninatype)
    #rank pockets
    class_model=Model()
    class_checkpoint=torch.load(args.class_checkpoint)
    types_lines=open(class_types,'r').readlines()
    batch_size = len(types_lines)
    #avoid cuda out of memory
    if batch_size>50:
        batch_size=50
    class_model, class_gmaker, class_eptest=get_model_gmaker_eprovider(class_types,batch_size,class_model,class_checkpoint)
    #divisible by 50 if types_lines > 50
    class_labels, class_probs = test_model(class_model, class_eptest, class_gmaker,  batch_size)
    zipped_lists = zip(class_probs[:len(types_lines)], types_lines)
    sorted_zipped_lists = sorted(zipped_lists,reverse=True)
    # print(sorted_zipped_lists)
    ranked_types = [element for _, element in sorted_zipped_lists]
    seg_types= class_types.replace('.types','_ranked.types')
    fout=open(seg_types,'w')
    fout.write(''.join(ranked_types))
        # fout.write(''.join(ranked_types[]))
    fout.close()
    # create pocket_locations.txt
    pocket_locations_file=os.path.join(fpocket_dir,'pocket_locations.csv')
    ranked_pockets = [(element, _)  for _, element in sorted_zipped_lists]
    fout=open(pocket_locations_file,'w')

    for pocket in ranked_pockets:
        coordinates = pocket[0].split()
        fout.write(str(coordinates[1])+','+str(coordinates[2])+','+str(coordinates[3])+','+str(pocket[1])+'\n')
    fout.close()
    del class_model
    del class_checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    #segmentation
    if args.rank!=0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model = Unet(args.num_classes, args.upsample)
        seg_model.to(device)
        seg_checkpoint = torch.load(args.seg_checkpoint)
        seg_model = nn.DataParallel(seg_model)
        seg_model, seg_gmaker, seg_eptest=get_model_gmaker_eprovider(seg_types,1,seg_model,seg_checkpoint,dims=32)
        dx_name=protein_nowat_file.replace('.pdb','')
        test(seg_model, seg_eptest, seg_gmaker,device,dx_name, args)
        del seg_model
        del seg_checkpoint
# /api/v1/rank?protein=protein.pdb&num_pockets=10 - rank pockets
@app.post(BASE_API + "/rank/")
async def rank_pockets_n( file: UploadFile = File(...)):
    '''
    Rank pockets for a protein
    '''
    try:
        # save protein to disk
        with open('protein.pdb', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        # command to run deeppocket
        
        rank_pockets('protein.pdb', RANK_MODEL, SEG_MODEL)

        # store pocket_locations.csv in memory
        with open('protein_nowat_out/pockets/pocket_locations.csv', 'rb') as f:
            pocket_locations = f.read()
        # return pocket_locations as a csv file
        # read ranked pockets

        # delete protein and ranked pockets
        os.remove('protein.pdb')
        # delete all pdb files in the current directory 
        for file in os.listdir():
            if file.endswith('.pdb'):
                os.remove(file)
        # delete all files and directories in protein_nowat_out directory
        for file in os.listdir('./protein_nowat_out'):
            if os.path.isdir('./protein_nowat_out/'+file):
                shutil.rmtree('./protein_nowat_out/'+file)
            else:
                os.remove('./protein_nowat_out/'+file)
    except Exception as e:
        ranked_pockets = 'No pockets found/ Invalid protein, training files'
        try: 
            os.remove('protein.pdb')
            # delete all pdb files in the current directory 
            for file in os.listdir():
                if file.endswith('.pdb'):
                    os.remove(file)
            # delete all files and directories in protein_nowat_out directory
            for file in os.listdir('./protein_nowat_out'):
                if os.path.isdir('./protein_nowat_out/'+file):
                    shutil.rmtree('./protein_nowat_out/'+file)
                else:
                    os.remove('./protein_nowat_out/'+file)
        except:
            pass
        return ranked_pockets
    return StreamingResponse(io.BytesIO(pocket_locations), media_type="text/csv", headers={"Content-Disposition": "attachment;filename=pocket_locations.csv"})


# /api/v1/segment?protein=protein.pdb&num_pockets=10 - segment pockets
@app.post(BASE_API + "/segment/num_pockets/{num_pockets}")
async def segment_pockets_n( file: UploadFile = File(...), num_pockets: int = 10):
    '''
    Segment pockets for a protein
    '''
    # save protein to disk
    try:
        with open('protein.pdb', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        # command to run deeppocket
        rank_pockets('protein.pdb', RANK_MODEL, SEG_MODEL, num_pockets)
        
        # get all .dx files from current directory and zip them
        files = os.listdir()
        dx_files = [f for f in files if f.endswith('.dx')]
        pdb_files = [f for f in files if f.endswith('.pdb')]
        if len(dx_files) == 0:
            return 'No pockets found'
        with ZipFile('pockets.zip', 'w') as zipObj:
            # Write protein_nowa_out/pockets/pocket_locations.csv as pockets_locations.csv to the zip
            zipObj.write('protein_nowat_out/pockets/pocket_locations.csv', 'pocket_locations.csv')
            for f in dx_files:
                zipObj.write(f)
            for f in pdb_files:
                zipObj.write(f)
            zipObj.write('readme.md')

        # delete all .pdb and .dx and .gninatypes files in the current directory
        for file in os.listdir():
            if file.endswith('.pdb') or file.endswith('.dx') or file.endswith('.gninatypes'):
                os.remove(file)
        # delete all files and directories in protein_nowat_out directory
        for file in os.listdir('./protein_nowat_out'):
            if os.path.isdir('./protein_nowat_out/'+file):
                shutil.rmtree('./protein_nowat_out/'+file)
            else:
                os.remove('./protein_nowat_out/'+file)
        # return pocket.zip and text file with pocket locations
        return  StreamingResponse(io.BytesIO(open('pockets.zip', 'rb').read()), media_type="application/zip", headers={"Content-Disposition": "attachment;filename=pockets.zip"} )
    except Exception as e:
        print(e)
        resp = 'No pockets found/ Invalid protein'
        try:
            for file in os.listdir():
                if file.endswith('.pdb') or file.endswith('.dx') or file.endswith('.gninatypes'):
                    os.remove(file)
            # delete all files and directories in protein_nowat_out directory
            for file in os.listdir('./protein_nowat_out'):
                if os.path.isdir('./protein_nowat_out/'+file):
                    shutil.rmtree('./protein_nowat_out/'+file)
                else:
                    os.remove('./protein_nowat_out/'+file)
            resp = 'No pockets found/ Invalid protein'
        except:
            pass
    return resp

if __name__ == "__main__":
    uvicorn.run(app, host = "localhost" ,port=8000)
