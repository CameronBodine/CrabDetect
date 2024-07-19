
#============================================
# Parameters
project_mode = 1
inDir = r'/home/cbodine/Desktop/CrabPot_InferenceTest_20240709/recordings'
outDir = r'/home/cbodine/Desktop/CrabPot_InferenceTest_20240709/crabdetect'
prefix = 'CrabPotTest_'
suffix = ''
aoi = r'/home/cbodine/Desktop/CrabPot_InferenceTest_20240709/recordings/Rec00002/PINGMapper_aoi/PINGMapper_aoi_coverage.shp'

cropRange = 0


#============================================

# Imports
import sys, os
pingPath = os.path.normpath('../PINGMapper/src')
sys.path.insert(0, pingPath)
sys.path.insert(0, 'src')

from funcs_common import *
from main_readFiles import read_master_func
from main_rectify import rectify_master_func
from main_crabDetect import crabpots_master_func

from glob import glob

#============================================

# Get processing script's dir so we can save it to file
scriptDir = os.getcwd()

# For the logfile
oldOutput = sys.stdout

# For the logfile
logfilename = 'log_'+time.strftime("%Y-%m-%d_%H%M")+'.txt'

#============================================
# PINGMapper Parameters
params = {
    'project_mode': project_mode,
    'aoi': aoi,
    'cropRange': cropRange,
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
inFiles = [inFiles[1]]

for i, f in enumerate(inFiles):
    print(i, ":", f)

for datFile in inFiles:
    logfilename = 'log_'+time.strftime("%Y-%m-%d_%H%M")+'.txt'

     
    # try:
    copied_script_name = os.path.basename(__file__).split('.')[0]+'_'+time.strftime("%Y-%m-%d_%H%M")+'.py'
    script = os.path.join(scriptDir, os.path.basename(__file__))

    start_time = time.time()  

    inPath = os.path.dirname(datFile)
    humFile = datFile
    recName = os.path.basename(humFile).split('.')[0]
    sonPath = humFile.split('.DAT')[0]
    # sonFiles = sorted(glob(sonPath+os.sep+'*.SON'))
    # Only need port and star
    sonFiles = glob(sonPath+os.sep+'B002.SON') + glob(sonPath+os.sep+'B003.SON')

    recName = prefix + recName + suffix

    projDir = os.path.join(outDir, recName)

    # =========================================================
    # Determine project_mode
    print(project_mode)
    if project_mode == 0:
        # Create new project
        if not os.path.exists(projDir):
            os.mkdir(projDir)
        else:
            projectMode_1_inval()

    elif project_mode == 1:
        # Overwrite existing project
        if os.path.exists(projDir):
            shutil.rmtree(projDir)

        os.mkdir(projDir)        

    elif project_mode == 2:
        # Update project
        # Make sure project exists, exit if not.
        
        if not os.path.exists(projDir):
            projectMode_2_inval()

    # =========================================================
    # For logging the console output

    logdir = os.path.join(projDir, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logfilename = os.path.join(logdir, logfilename)

    sys.stdout = Logger(logfilename)

    print('\n\n', '***User Parameters***')
    for k,v in params.items():
        print("| {:<20s} : {:<10s} |".format(k, str(v)))

    #============================================
    # Add ofther params
    params['sonFiles'] = sonFiles
    params['logfilename'] = logfilename
    params['script'] = [script, copied_script_name]
    params['projDir'] = projDir
    params['humFile'] = humFile



    print('sonPath',sonPath)
    print('\n\n\n+++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('***** Working On *****')
    print(humFile)
    print('Start Time: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

    print('\n===========================================')
    print('===========================================')
    print('***** READING *****')
    read_master_func(**params)

    print('\n===========================================')
    print('===========================================')
    print('***** RECTIFYING *****')
    rectify_master_func(**params)

    print('\n===========================================')
    print('===========================================')
    print('***** DETECTING CRAB POTS *****')
    crabpots_master_func(**params)    
        
    # except Exception as Argument:
    #     unableToProcessError(logfilename)
    #     print('\n\nCould not process:', datFile)


        