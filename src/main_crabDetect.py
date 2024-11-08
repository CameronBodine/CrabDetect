'''

'''

from funcs_common import *
from fiona.drvsupport import supported_drivers
supported_drivers['KML'] = 'rw'
supported_drivers['kml'] = 'rw'
supported_drivers['libkml'] = 'rw'
supported_drivers['LIBKML'] = 'rw'

from class_crabObj import crabObj

#===========================================
def crabpots_master_func(logfilename='',
                        project_mode=0,
                        script='',
                        humFile='',
                        sonFiles='',
                        projDir='',
                        coverage=False,
                        aoi=False,
                        tempC=10,
                        nchunk=500,
                        cropRange=0,
                        exportUnknown=False,
                        fixNoDat=False,
                        threadCnt=0,
                        pix_res_son=0,
                        pix_res_map=0,
                        x_offset=0,
                        y_offset=0,
                        tileFile=False,
                        egn=False,
                        egn_stretch=0,
                        egn_stretch_factor=1,
                        wcp=False,
                        wcr=False,
                        lbl_set=False,
                        spdCor=0,
                        maxCrop=False,
                        USE_GPU=False,
                        remShadow=0,
                        detectDep=0,
                        smthDep=0,
                        adjDep=0,
                        pltBedPick=False,
                        rect_wcp=False,
                        rect_wcr=False,
                        son_colorMap='Greys',
                        pred_sub=0,
                        map_sub=0,
                        export_poly=False,
                        map_predict=0,
                        pltSubClass=False,
                        map_class_method='max',
                        mosaic_nchunk=50,
                        mosaic=False,
                        map_mosaic=0,
                        banklines=False,
                        gpxToHum=True,
                        sdDir='',
                        threshold=0.5,
                        wptPrefix=''):
    '''
    '''

    start_time = time.time()

    ############
    # Parameters
    # Specify multithreaded processing thread count
    if threadCnt==0: # Use all threads
        threadCnt=cpu_count()
    elif threadCnt<0: # Use all threads except threadCnt; i.e., (cpu_count + (-threadCnt))
        threadCnt=cpu_count()+threadCnt
        if threadCnt<0: # Make sure not negative
            threadCnt=1
    else: # Use specified threadCnt if positive
        pass

    if threadCnt>cpu_count(): # If more than total avail. threads, make cpu_count()
        threadCnt=cpu_count();
        print("\nWARNING: Specified more process threads then available, \nusing {} threads instead.".format(threadCnt))

    ####################################################
    # Check if sonObj pickle exists, append to metaFiles
    metaDir = os.path.join(projDir, "meta")
    print(metaDir)
    if os.path.exists(metaDir):
        metaFiles = sorted(glob(metaDir+os.sep+"*.meta"))
    else:
        sys.exit("No SON metadata files exist")
    del metaDir

    #############################################
    # Create a crabObj instance from pickle files
    crabObjs = []
    for meta in metaFiles:
        son = crabObj(meta) # Initialize mapObj()
        crabObjs.append(son) # Store mapObj() in mapObjs[]
    del meta, metaFiles

    ########################
    # For Crab Pot Detection
    ########################

    api_key = 'w9qOuYiN7EpEMqAYEln2'
    model_name = 'allcrabpotsources'
    # model_version = '7'
    # model_version = '8'
    model_version = '10'

    detectCrabPot = True
    if detectCrabPot:
        start_time = time.time()

        print('\n\nAutomatically detecting crab pots...')

        outDir = os.path.join(crabObjs[0].projDir, 'detect_CrabPots')
        if not os.path.exists(outDir):
            os.mkdir(outDir)

        # Get chunk id for mapping substrate
        for son in crabObjs:
            # Set outDir
            son.outDir = outDir

            # Get chunk id's
            chunks = son._getChunkID()

            # Prepare model
            son.crabModel_id = '{}/{}'.format(model_name, model_version)
            son.crabModel_api = api_key

            # Do prediction (make parallel later)
            print('\n\tDetecting crab pots for', len(chunks), son.beamName, 'chunks')

            # r = Parallel(n_jobs=np.min([len(chunks), threadCnt]), verbose=10)(delayed(son._detectCrabPots_moveWin)(i) for i in chunks)
            r = Parallel(n_jobs=np.min([len(chunks), threadCnt]), verbose=10)(delayed(son._detectCrabPots)(i) for i in chunks)
            # for i in chunks:
            #     son._detectCrabPots(i)

            # for i in chunks:
            #     son._detectCrabPots_moveWin(i)

            # Remove None instances
            # r = r.remove(None)
            r = [i for i in r if i is not 0]


            if len(r)>0:
                df = pd.concat(r)
                
                if 'dfDetect' not in locals():
                    dfDetect = df
                else:
                    dfDetect = pd.concat([dfDetect, df], ignore_index=True)

        if 'dfDetect' in locals():

            # Save predictions to csv
            projName = os.path.split(son.projDir)[-1]
            file_name = projName + '_detect_results.csv'
            file_name = os.path.join(son.outDir, file_name)

            dfDetect.to_csv(file_name, index=False)
            del dfDetect

            # For testing
            # dfDetect = pd.read_csv(file_name)

            for son in crabObjs:
                son.crabDetectCSV = file_name
                son._pickleSon()

            print('\n\nCalculating crabpot coordinates...')

            for son in crabObjs:
                r = son._calcDetectCoords()

                if not 'dfDetect' in locals():
                    dfDetect = r
                else:
                    dfDetect = pd.concat([dfDetect, r])

            dfDetect.reset_index(drop=True, inplace=True)

            # Update name
            # Calculate name
            for i, row in dfDetect.iterrows():
                # conf = int(row['confidence']*100)
                # if row['class_name'] == 'Crab-Pot':
                #     class_name = 'CP'
                # elif row['class_name'] == 'Maybe-Pot':
                #     class_name = 'MCP'
                # else:
                #     class_name = 'T'
                # wptName = '{} {} {}%'.format(class_name, i, conf)
                # dfDetect.loc[i, 'name'] = wptName

                if i < 10:
                    zeros = '00'
                elif i < 100:
                    zeros = '0'
                else:
                    zeros = ''
                wptName = '{}_{}{}'.format(wptPrefix, zeros, i)
                dfDetect.loc[i, 'name'] = wptName

            # Open as geopandas dataframe
            gdf = gpd.GeoDataFrame(dfDetect, geometry=gpd.points_from_xy(dfDetect['pot_lon'], dfDetect['pot_lat']), crs="EPSG:4326")
            
            # Save as kml
            file_name = os.path.split(son.projDir)[-1] + '.kml'
            file_name = os.path.join(son.outDir, file_name)

            gdf.to_file(file_name, driver='KML')

            # Save as shapefile
            file_name = file_name.replace('.kml', '.shp')
            gdf.to_file(file_name)

            if gpxToHum:

                # Calculate Humminbird Waypoints
                crabObjs[0]._calcHumWpt(sdDir, threshold)

        else:
            print('\n\nNo crab pots detected. Bye Bye!')

    print("\n\nTotal Detection Time: ",datetime.timedelta(seconds = round(time.time() - start_time, ndigits=0))) 
            

        


