import json

import numpy as np
from scipy.spatial import ConvexHull

cocojson = 'annotations/trans_drone_rough_annotations.json'
destLabels = "dota_test/" #directory where DOTA label files should be stored.
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval



f = open(cocojson,)

data = json.load(f)
 
# Iterating through the json
# list
dict_cat = {}
for i in data['categories']:
    dict_cat[i["id"]] = i["name"].replace(" ", "-")
print("Category dict:")
print("\n"+str(dict_cat) +"\n")

print("\nKeys in data:")
for key, val in data.items():
    print(key)
c = 0
#for each image get image_id, and use that to obtain annotations and write to file
for img in data["images"]:
    img_id = img["id"] #int
    file_name = img["file_name"][0:-4] #str, without the extension, extension is 4 char, .txt
    with open(destLabels+file_name+".txt", 'w') as fileDOTA:
        fileDOTA.write('imagesource:DJI\ngsd:None')
    for ann in data["annotations"]:
        if ann["image_id"] == img_id:
            cat = dict_cat[ann["category_id"]] #str: get category of current annotation.
            points = np.array(ann["segmentation"][0]).reshape(-1, 2) #reshape seg points in 2d np array
            bbox = minimum_bounding_rectangle(points) #get rotated bbox
            #convert bbox to INT. Important
            bbox = bbox.astype(int)
            bbox[bbox<0] = 0 #replacing all negative values with 0. Just to be safe.
            #print(bbox)
            #TODO: check if len bbox == 0, continue.
            with open(destLabels+file_name+".txt", 'a') as fileDOTA:
                line = ' '.join(' '.join(str(x) for x in y) for y in bbox)
                fileDOTA.write("\n"+line + " "+ cat + " "+ "0")
    #remove img_id from data["images"]


'''with open('dota_test/a.txt', 'a') as the_file:
    the_file.write('imagesource:DJI\ngsd:None\n')'''
#x = data['annotations'][0]['segmentation']
#print(x[0])
#print(x)
f.close()
print("\nAll done!")