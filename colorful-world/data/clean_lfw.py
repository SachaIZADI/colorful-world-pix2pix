import os
FOLDER = 'lfw/'
list_subfolder = os.listdir(FOLDER)
for subfold in list_subfolder:
    for file in os.listdir(FOLDER+subfold):
        os.rename(FOLDER+subfold+"/"+file,FOLDER+file)
    os.rmdir(FOLDER+subfold)