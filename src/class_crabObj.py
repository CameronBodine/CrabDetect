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

            # If multiple predictions in an image
            if len(df) > 1:
                df['chunk'] = i
                df['beam'] = self.beamName
                df['name'] = os.path.basename(file_name)
                df['img_width'] = df.loc[0, 'img_width']
                df['img_height'] = df.loc[0, 'img_height']

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

            if len(trk) > 0:

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
    

    #=======================================================================
    def _calcHumWpt(self):
        # For Waypoint Name
        namePrefix = 'Pot '

        # Get predictions
        predDF = pd.read_csv(self.crabDetectCSV)

        # Calculate name
        for i, row in predDF.iterrows():
            zero = self._addZero(i)
            wptName = namePrefix+'{}{}'.format(zero, i)
            predDF.loc[i, 'wpt_name'] = wptName

        # Save to gpx
        # gdf = gpd.GeoDataFrame(predDF, geometry=gpd.points_from_xy(predDF['pot_lon'], predDF['pot_lat']), crs="EPSG:4326")
        gdf = predDF[['wpt_name', 'pot_lat', 'pot_lon']]
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(predDF['pot_lon'], predDF['pot_lat']), crs='EPSG:4326')
        gdf = gdf.rename(columns={'wpt_name': 'name'})
        gdf = gdf[['name', 'geometry']]

        file_name = os.path.join(self.outDir, 'CrabPotLoc.gpx')
        gdf.to_file(file_name, 'GPX')

        # Save df
        predDF.to_csv(self.crabDetectCSV, index=False)
    

    #=======================================================================
    def _calcHumWpt_old(self):
        # For Waypoint Name
        namePrefix = 'Pot '

        # Get predictions
        predDF = pd.read_csv(self.crabDetectCSV)
    
        # Configure re-projection function
        epsg = 'EPSG:3395' # World mercator
        trans = pyproj.Proj(epsg)
        
        e, n = trans(predDF['pot_lon'], predDF['pot_lat'])
        predDF['e'] = (np.round(e,0)).astype('int')
        predDF['n'] = (np.round(n,0)).astype('int')

        # Calculate name
        for i, row in predDF.iterrows():
            zero = self._addZero(i)
            wptName = namePrefix+'{}{}'.format(zero, i)
            predDF.loc[i, 'wpt_name'] = wptName

        # Wpt header (FIRST)
        header = []
        pnt_head = 33685540 # Each point starts with this
        pnt_head = pnt_head.to_bytes(4, 'big')
        header.append(pnt_head)
        
        spacer = 0
        head_2 = spacer.to_bytes(4, 'big')
        header.append(head_2)

        head_3 = 268435456
        head_3 = head_3.to_bytes(4, 'big')
        header.append(head_3)

        head_4 = head_2 
        header.append(head_4)

        head_5 = 4294901760
        head_5 = head_5.to_bytes(4, 'big')
        header.append(head_5)

        head_6 = 3221291008
        head_6 = head_6.to_bytes(4, 'big')
        header.append(head_6)

        head_7 = 'Home'
        header.append(head_7)

        head_8 = head_2
        header.append(head_8)

        head_9 = head_2
        header.append(head_9)


        # Write header to file
        file_name = os.path.join(self.outDir, 'DATA.HWR')
        # Delete file if it exists
        if os.path.exists(file_name):
            os.remove(file_name)

        for h in header:
            try:
                file = open(file_name, 'ab')
                file.write(h)
                file.close()
            except:
                file = open(file_name, 'a')
                file.write(h)
                file.close()

        # WPT Header (SECOND)
        header = []
        pnt_number = 1
        pnt_number = pnt_number.to_bytes(2, 'big')

        pnt_spacer = 0
        pnt_spacer = pnt_spacer.to_bytes(2, 'big')

        spacer = 285261838
        spacer = spacer.to_bytes(4, 'big')

        header.append(pnt_head)
        header.append(pnt_number)
        header.append(pnt_spacer)
        header.append(spacer)

        for v in np.arange(2, 14, 1):
            v = int(v)
            v = v.to_bytes(2, 'big')
            header.append(v)

        for h in header:
            file = open(file_name, 'ab')
            file.write(h)
            file.close()



        # Iterate predictions and add to Humminbird file

        for i, row in predDF.iterrows():
            # Get values
            lat = row['n'].to_bytes(4, 'big', signed=True)
            lon = row['e'].to_bytes(4, 'big', signed=True)
            wpt_name = row['wpt_name']
            i += 2
            pnt_number = i.to_bytes(2, 'big')
            
            spacer = 0
            spacer = spacer.to_bytes(2, 'big')

            symbol = 1 # ??? Waypoint symbol, I think
            symbol = symbol.to_bytes(4, 'big')
            unknown = 1721152245
            unknown = unknown.to_bytes(4, 'big')

            # Prep name and spacer
            if len(wpt_name) < 12:
                to_pad = 12 - len(wpt_name)
                zero = 0
                wpt_name_spacer = zero.to_bytes(to_pad, 'big')
            elif len(wpt_name > 12):
                # Must slice name, too long
                old_name = wpt_name.copy()
                wpt_name = wpt_name[:12]
                print('Waypoint name ({}) is too long. Trimming to "{}"'.format(old_name, wpt_name))
                wpt_name_spacer = ''
            else:
                wpt_name_spacer = ''

            wpt = [pnt_head, pnt_number, spacer, symbol, unknown, lon, lat, wpt_name, wpt_name_spacer]

            for w in wpt:
                if isinstance(w, str):
                    file = open(file_name, 'a')
                    file.write(w)
                    file.close()
                else:
                    file = open(file_name, 'ab')
                    file.write(w)
                    file.close()

        # Add tail
        tail_1 = 196542464
        tail_1 = tail_1.to_bytes(4, 'big')

        tail_2 = -1862205441
        tail_2 = tail_2.to_bytes(4, 'big', signed=True)

        tail_end = -1
        tail_end = tail_end.to_bytes(4, 'big', signed=True)

        tail = [pnt_head, tail_1, tail_2, tail_end, tail_end, tail_end, tail_end, tail_end, tail_end]

        for t in tail:
            file = open(file_name, 'ab')
            file.write(t)
            file.close()

        # Save to gpx
        # gdf = gpd.GeoDataFrame(predDF, geometry=gpd.points_from_xy(predDF['pot_lon'], predDF['pot_lat']), crs="EPSG:4326")
        gdf = predDF[['wpt_name', 'pot_lat', 'pot_lon']]
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(predDF['pot_lon'], predDF['pot_lat']), crs='EPSG:4326')
        gdf = gdf.rename(columns={'wpt_name': 'name'})
        gdf = gdf[['name', 'geometry']]

        file_name = os.path.join(self.outDir, 'CrabPotLoc.gpx')
        gdf.to_file(file_name, 'GPX')

        # Save df
        predDF.to_csv(self.crabDetectCSV, index=False)

        

        

        


    


