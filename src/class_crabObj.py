'''
'''

from funcs_common import *
from class_rectObj import rectObj

import inference
import supervision as sv
import cv2
import json

class crabObj(rectObj):

    '''
    '''

    ############################################################################
    # Create crabObj() instance from previously created rectObj() instance     #
    ############################################################################

    #=======================================================================
    def __init__(self,
                 metaFile):

        rectObj.__init__(self, metaFile)

        return
    
    #=======================================================================
    def _detectCrabPots(self, i, export_image=True):

        # Get the model
        model = inference.get_model(model_id=self.crabModel_id,
                                     api_key=self.crabModel_api)

        # Get sonMeta df
        if not hasattr(self, "sonMetaDF"):
            self._loadSonMeta()
        df = self.sonMetaDF

        # Get sonDat
        self._getScanChunkSingle(i)
        image = self.sonDat

        # Expecting three band, so stack
        image = np.dstack((image, image, image)).astype('float32')

        # Do inference
        results = model.infer(image)[0]
        preds = results.predictions

        if len(preds) > 0:

            # load the results into the supervision Detections api
            detections = sv.Detections.from_inference(results)

            # create supervision annotators
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # annotate the image with our inference results
            annotated_image = bounding_box_annotator.annotate(
                            scene=image, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections)
            
            channel = os.path.split(self.beamName)[-1] #ss_port, ss_star, etc.
            projName = os.path.split(self.projDir)[-1]
            file_name = projName + '_' + channel + '_detect_results_' + self._addZero(i) + str(i) + '.png'
            file_name = os.path.join(self.outDir, file_name)
            
            if export_image:
                cv2.imwrite(file_name, annotated_image)

            results = results.json()
            results = json.loads(results)

            # Prepare dataframe
            df = pd.DataFrame.from_dict({'chunk':[i], 'beam':[self.beamName], 'name':[os.path.basename(file_name)]})
            df1 = pd.json_normalize(results['image'])
            df1 = df1.rename(columns={'width': 'img_width', 'height': 'img_height'})
            df2 = pd.json_normalize(results['predictions'])

            df = pd.concat([df, df1, df2], axis=1)

            return df
        else:
            return

    #=======================================================================
    def _calcDetectCoords(self,
                          flip=False,
                          wgs=False,
                          cog=True):
        
        lons = 'trk_lons'
        lats = 'trk_lats'
        ping_bearing = 'ping_bearing'

        # Get predictions
        predDF = pd.read_csv(self.crabDetectCSV)

        # Filter by beam
        predDF = predDF[predDF['beam'] == self.beamName].reset_index(drop=True)

        # Get smoothed trackline
        trkMetaFile = os.path.join(self.metaDir, "Trackline_Smth_"+self.beamName+".csv")
        sDF = pd.read_csv(trkMetaFile)

        ########################
        # Calculate ping bearing
        # Determine ping bearing.  Ping bearings are perpendicular to COG.
        if self.beamName == 'ss_port':
            rotate = -90  # Rotate COG by 90 degrees to the left
        else:
            rotate = 90 # Rotate COG by 90 degrees to the right
        if flip: # Flip rotation factor if True
            rotate *= -1

        # Calculate ping bearing and normalize to range 0-360
        # cog = False
        if cog:
            sDF[ping_bearing] = (sDF['trk_cog']+rotate) % 360
        else:
            sDF[ping_bearing] = (sDF['instr_heading']+rotate) % 360

        # Use WGS 1984 coordinates and set variables as needed
        if wgs is True:
            epsg = self.humDat['wgs']
            xRange = 'range_lons'
            yRange = 'range_lats'
            xTrk = 'trk_lons'
            yTrk = 'trk_lats'
        ## Use projected coordinates and set variables as needed
        else:
            epsg = self.humDat['epsg']
            xRange = 'range_es'
            yRange = 'range_ns'
            xTrk = 'trk_utm_es'
            yTrk = 'trk_utm_ns'

        # Pixel size (in meters)
        pix_m = self.pixM

        # Iterate each crab pot
        for i, pot in predDF.iterrows():
            # Determine which record in smoothed trackline from:
            ## Chunk +
            ## Offset (x)
            potChunk = pot['chunk']
            potX = int(pot['x'])
            
            # Filter smoothed trackline by chunk
            trk = sDF[sDF['chunk_id'] == potChunk].reset_index(drop=True)

            # Filter smoothed trackline by ping
            trk = trk.filter(items=[potX], axis=0)

            # Determine distance based on:
            ## y and
            ## pix_m
            potY = int(pot['y'])
            d = potY * pix_m


            # Calculate the coordinates from:
            ## origin (track x/y), distance, and COG
            R = 6371.393*1000 #Radius of the Earth in meters
            brng = np.deg2rad(trk[ping_bearing].values[0])

            # Get lat/lon for origin of each ping
            lat1 = np.deg2rad(trk[lats].values[0])#.to_numpy()
            lon1 = np.deg2rad(trk[lons].values[0])#.to_numpy()

            # Calculate latitude of range extent
            lat2 = np.arcsin( np.sin(lat1) * np.cos(d/R) +
                np.cos(lat1) * np.sin(d/R) * np.cos(brng))

            # Calculate longitude of range extent
            lon2 = lon1 + np.arctan2( np.sin(brng) * np.sin(d/R) * np.cos(lat1),
                                    np.cos(d/R) - np.sin(lat1) * np.sin(lat2))

            # Convert range extent coordinates into degrees
            lat2 = np.degrees(lat2)
            lon2 = np.degrees(lon2)

            # Store in df
            predDF.loc[i, 'pot_lat'] = lat2
            predDF.loc[i, 'pot_lon'] = lon2

        return predDF

