import csv
import scipy.io as sio
outputdir = '../data/archive01/food_label.mat'
filedir = '../data/archive01/food_label.csv'
image_name = []
image_label = []
with open(filedir) as f_handle:
    reader = csv.reader(f_handle)
    for row in reader:
        if row[0] == 'instagram_id':
            continue
        image_name.append(row[0])
        image_label.append(int(row[1]))

sio.savemat(outputdir, {'im_name':image_name, 'im_label':image_label})

