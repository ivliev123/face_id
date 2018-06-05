# python3 make_id.py  -f 'da.pickle' -m 'w' -t 'ID.txt'

import pickle
import argparse
import dlib
from skimage import io
from scipy.spatial import distance


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filefrom",type=str, default="",
	help="использование pi камеры")
ap.add_argument("-t", "--fileto", type=str, default="ID.txt",
	help="выбор файла")
ap.add_argument("-m", "--metod", type=str, default="w",
	help="выбор- открытие файла для чтения, записи, дозаписи")
args = vars(ap.parse_args())


ID=0

with open(args["filefrom"], "rb") as f:
    array_data = []
    try:
        while True:
            array_data.append(pickle.load(f))
    except (EOFError, pickle.UnpicklingError):
        pass

print(array_data)

for i in range(len(array_data)):
	if (i==0):
		array_data[i][1]=0
		ID=ID+1

	if (array_data[i][1]=="none"):
		array_data[i][1]=ID
		ID=ID+1

	for k in range(len(array_data)-i-1):
		if (array_data[k+i+1][1]=="none"):
			dist=distance.euclidean(array_data[i][4], array_data[k+i+1][4])
			array_data[i+k+1][5]=dist

			
			if (dist<=0.5 	):
				array_data[k+i+1][1]=array_data[i][1]
				print('ID',array_data[k+i+1][1])

d = open(args["fileto"], args["metod"])
for i in range(len(array_data)):
	s= str(array_data[i][1]) +"  "+ array_data[i][2] +"  " + array_data[i][3] 
	d.write(s+ '\n')


d.close()
