
#============================================
# Parameters
inDir = r'C:\Users\bodin\Desktop\CrabPot_InferenceTest_20240709\recordings'
prefix = 'CrabPotTest'
suffix = ''

#============================================

# Imports
import sys, os
sys.path.insert(0, 'PINGMapper/src')

from funcs_common import *
from main_readFiles import read_master_func
from main_rectify import rectify_master_func

from glob import glob

#============================================

# For the logfile
logfilename = 'log_'+time.strftime("%Y-%m-%d_%H%M")+'.txt'

#============================================
# PINGMapper Parameters
params = {

}

globals().update(params)

#============================================

# Find all DAT and SON files in all subdirectories of inDir
inFiles=[]
for root, dirs, files in os.walk(inDir):
    for file in files:
        if file.endswith('.DAT'):
            inFiles.append(os.path.join(root, file))

inFiles = sorted(inFiles)

for i, f in enumerate(inFiles):
    print(i, ":", f)

for datFile in inFiles:
     logfilename = 'log_'+time.strftime("%Y-%m-%d_%H%M")+'.txt'
    
    try:
        copied_script_name = os.path.basename(__file__).split('.')[0]+'_'+time.strftime("%Y-%m-%d_%H%M")+'.py'
        script = os.path.join(scriptDir, os.path.basename(__file__))

        start_time = time.time()  

        inPath = os.path.dirname(datFile)
        humFile = datFile
        recName = os.path.basename(humFile).split('.')[0]
        sonPath = humFile.split('.DAT')[0]
        sonFiles = sorted(glob(sonPath+os.sep+'*.SON'))

        recName = prefix + recName + suffix

        projDir = os.path.join(outDir, recName)

    except Exception as Argument:
        unableToProcessError(logfilename)
        print('\n\nCould not process:', datFile)
        