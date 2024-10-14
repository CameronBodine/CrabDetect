
'''
Automated crab pot detection from Humminbird side imaging systems:

1) Extract sonar data from Humminbird .SON files (PING-Mapper)
2) Predict crab pots (Roboflow Inference Python API; model from Dr. T)
3) Calculate GPS coordinates (PING-Mapper)
'''

#============================================
# Parameters

# inDir = r'/media/cbodine/UDEL_Ubuntu/SynologyDrive/UDEL/Bootcamp/2024/CrabPots/BoatBasinTest/sss_raw'
# outDir = r'/media/cbodine/UDEL_Ubuntu/SynologyDrive/UDEL/Bootcamp/2024/CrabPots/BoatBasinTest/MoveWinTest'

# project_mode = 1
# prefix = 'InferanceTest_'
# suffix = ''
# aoi = r'/home/cbodine/Desktop/CrabPot_InferenceTest_20240709/recordings/Rec00002/PINGMapper_aoi/PINGMapper_aoi_coverage.shp'
# gpxToHum = True
# threshold = 0.5

# nchunk = 500
# cropRange = 25
# rect_wcp = False


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

# GUI
PySimpleGUI_License = 'e3yAJ9MVaOWANplCbmndNNl2VwHvlCwpZTSjIl6DIjkiRGpYc53aRty8aBWpJF1qdwGLlzv9bUiHILs3Inkyxpp5Yq2OVku8cg2ZVrJ7RNCQI66bMcTLcnyKMbTRMK57OCTPMGxGNtS8whirTBGTlLjxZEWg5DzWZdUXRUlLcDGfxnv7eiWB1jlOb6nqR8WTZ2XsJVzbabW19ouWI6j0oXiKN0Si4AwtI7iFw8iGTBmtFftjZEUxZMpYcLncNk0rIJj4oyisQq2uFCtqZnXWJvvqbEiCICsSIbkC5jhKbvWTVqM2YtX6Ni0XIJjloji1QEmU9Ak5ayWp5nlnIwi3wiiOQK279ytqcKGwFGuvepS6IH6iIOiYIGs7I4kYNQ13cY33RkvIbqWkVyypSKUOQoiZO2ijIFzaMNTEAp0bNxyWI1sLIwkRRjhZdNGBVoJkcZ3MNN1yZMWTQtihOiieIYyXMDDIIF0ILaTyAt36LKTREj5JI1iYwcixRgGuFk0BZGU5VZ4dciGUl3ykZ3XtMbilOMiBIhy1M5DtId1mL6T1A935LYTLEN5iI3iJwoirR8Wa12h5a0WtxkBNZdGiRJyYZXX9N5zZI2jSoZizYpmp9YkHaIWz5YluLTmcNXzNQmGZd0twYGW6l3sALZmTNWvubcSEItsPITk6lFQgQUWZRrkfcEmAVxz0c9y7IG6sILjZEYyzO8Cf4c0WLDj3QCwSLwjPEt2BMMi0J69p854e39898f71ea82d3a530f7a6ed8a02a4eea9ffd2c7b1279074b491c71b411f392e6d726a2d2f9dbf63388356cf4e083e358fe428852d676073e128607b9ad194c15e34a4feb463a749fd3295606caa293b823d102e854cd845b79b5ec5eaec0b2ef7f9cf0c87b2dfcad3f14cd0d66a2da97e6b38a535eb8707b4486c9802a4bfeb09703382e157449096f0e3551af9f444197cacb3f3d42187cea97ab61978985ddeecd086b9cb86c4ec1c08082d47b3ed0ae9c044d9aa65e5c9bf6e00238f78ed858cfdaf0021fb95d636e0cce84d84d2c2da7ac57f2e54fe793fce44a8b8abf96ce7c381f4b7eeb55dc4b68768e8172a4dffc1b683e62a108b2dfc2ef340dab058e6ee5c1f525f93e89d39258862f099987a8ec7022db5aecb5a58e81d02370d5717d18498ae58749aa5e463cf757ab7fa84efe49c1b770da397eef22423696ad433e7232646e279906bef084b21714ac5fc2af564a03ebc789123aed44531765b3e72c6165131feab68e35e0276a64760ee9abf043bece1e3cd148bcec97ab835395391387ff9d2b74a835a15ea5bac9c7e1218c217481a3999a91e037a138aaf5dddadb2247141242140b130e273aab5e1e6855fae8b7ee80d64be2d09a46f3d49555f53a7a849138fc3b9d2323658ea7e86a0039c40f3c15fd3647f99ec98232d9734a5933177c48c6575a1415e2808640cfb27773e728fe128b99757'
import PySimpleGUI as sg

inDirInit = r'/media/cbodine'
outDirInit = r'/media/cbodine/UDEL_Ubuntu/SynologyDrive/UDEL/Bootcamp/2024/CrabPots/Demo/'

# inDirInit = r'/home/cbodine/Desktop/CrabPotDemo/20241011_Test/'
# outDirInit = r'/home/cbodine/Desktop/CrabPotDemo/20241011_Test/'

prefixDefault = 'CrabPot_20241011Test_'
cropRangeDefault = 0
nchunk = 500
rect_wcp = False
egn=True
egn_stretch=1


layout = [
    [sg.Text('Path to SD Card Sonar Recordings')],
    [sg.In(key='inDir', size=(80,1)), sg.FolderBrowse(initial_folder=inDirInit)],
    [sg.Text('Output Folder')],
    [sg.In(key='outDir', size=(80,1)), sg.FolderBrowse(initial_folder=outDirInit)],
    [sg.Text('Project Name Prefix:', size=(20,1)), sg.Input(key='prefix', size=(10,1), default_text=prefixDefault), sg.VerticalSeparator(), sg.Text('Project Name Suffix:', size=(20,1)), sg.Input(key='suffix', size=(10,1))],
    [sg.Checkbox('Export Detections to Humminbird SD Card', key='gpxToHum', default=True), sg.VerticalSeparator(), sg.Text('Confidence Threshold', size=(20,1)),sg.Slider((0,1), key='threshold', default_value=0.5, resolution=0.01, tick_interval=0.5, orientation='horizontal')],
    [sg.Text('Crop Range:', size=(10,1)), sg.Input(key='cropRange', size=(10,1), default_text=cropRangeDefault)],
    [sg.Text('Position Corrections')],
    [sg.Text('Transducer Offset [X]:', size=(22,1)), sg.Input(key='x_offset', default_text=0.0, size=(10,1)), sg.VerticalSeparator(), sg.Text('Transducer Offset [Y]:', size=(22,1)), sg.Input(key='y_offset', default_text=0.0, size=(10,1))],
    [sg.Submit('Detect Crab Pots'), sg.Quit()]
]

# layout2 =[[sg.Column(layout, scrollable=True,  vertical_scroll_only=True, size_subsample_height=2)]]
layout2 = [[sg.Column(layout)]]
window = sg.Window('Detect Crab Pots', layout2, resizable=True)


while True:
    event, values = window.read()
    if event == "Quit" or event == 'Detect Crab Pots':
        break

window.close()

if event == "Quit":
    sys.exit()

# Get parameters from GUI
inDir = values['inDir']
outDir = values['outDir']
prefix = values['prefix']
suffix = values['suffix']
gpxToHum = values['gpxToHum']
threshold = values['threshold']
cropRange = int(values['cropRange'])

project_mode=1


#============================================

# Get processing script's dir so we can save it to file
scriptDir = os.getcwd()

# For the logfile
oldOutput = sys.stdout

# For the logfile
logfilename = 'log_'+time.strftime("%Y-%m-%d_%H%M")+'.txt'

#============================================

# Find all DAT and SON files in all subdirectories of inDir
inFiles=[]
for root, dirs, files in os.walk(inDir):
    for file in files:
        if file.endswith('.DAT') and 'Trash' not in root:
            inFiles.append(os.path.join(root, file))

inFiles = sorted(inFiles)
# inFiles = [inFiles[1]]

for i, f in enumerate(inFiles):
    print(i, ":", f)

for datFile in inFiles:
    logfilename = 'log_'+time.strftime("%Y-%m-%d_%H%M")+'.txt'

     
    # try:
    copied_script_name = os.path.basename(__file__).split('.')[0]+'_'+time.strftime("%Y-%m-%d_%H%M")+'.py'
    script = os.path.join(scriptDir, os.path.basename(__file__))

    start_time = time.time()


    #============================================

    inPath = os.path.dirname(datFile)
    humFile = datFile
    recName = os.path.basename(humFile).split('.')[0]

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

        print(projDir)
        os.mkdir(projDir)        

    elif project_mode == 2:
        # Update project
        # Make sure project exists, exit if not.
        
        if not os.path.exists(projDir):
            projectMode_2_inval()

    
    #============================================
    # Copy humminbird files to projDir
    recordDir = os.path.join(projDir, 'recording')
    if not os.path.exists(recordDir):
        os.mkdir(recordDir)

    datName = os.path.basename(humFile)
    datDest = os.path.join(recordDir, datName)

    shutil.copy(datFile, datDest)

    sonPath = humFile.split('.DAT')[0]
    sonDest = os.path.join(recordDir, os.path.basename(sonPath))
    
    shutil.copytree(sonPath, sonDest)

    humFile = datDest
    sonPath = sonDest

    # Only need port and star
    sonFiles = glob(sonPath+os.sep+'B002.SON') + glob(sonPath+os.sep+'B003.SON')
    

    # =========================================================
    # For logging the console output

    logdir = os.path.join(projDir, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logfilename = os.path.join(logdir, logfilename)

    sys.stdout = Logger(logfilename)

    # print('\n\n', '***User Parameters***')
    # for k,v in params.items():
    #     print("| {:<20s} : {:<10s} |".format(k, str(v)))

    #============================================

    params = {
        'nchunk': nchunk,
        'project_mode': project_mode,
        # 'aoi': aoi,
        'cropRange': cropRange,
        'rect_wcp': rect_wcp,
        'x_offset':float(values['x_offset']),
        'y_offset':float(values['y_offset']),
    }

    globals().update(params)

    #============================================
    # Add ofther params
    params['sonFiles'] = sonFiles
    params['logfilename'] = logfilename
    params['script'] = [script, copied_script_name]
    params['projDir'] = projDir
    params['humFile'] = humFile
    params['egn'] = egn
    params['egn_stretch'] = egn_stretch


    #============================================

    print('\n\n', '***User Parameters***')
    for k,v in params.items():
        print("| {:<20s} : {:<10s} |".format(k, str(v)))


    try:

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

        params['gpxToHum'] = gpxToHum
        params['sdDir'] = inDir
        params['threshold'] = threshold

        print('\n===========================================')
        print('===========================================')
        print('***** DETECTING CRAB POTS *****')
        crabpots_master_func(**params)    
        
    except Exception as Argument:
        unableToProcessError(logfilename)
        print('\n\nCould not process:', datFile)


        