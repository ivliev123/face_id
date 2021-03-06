# python3 make_id_finish6.py  -m 'w' -t 'ID.txt'

import pickle
import argparse
import dlib
from imutils import face_utils
from skimage import io
from scipy.spatial import distance
import numpy as np
import copy
import time
import cv2
import imutils
import os
import fnmatch

def index_min(array, n):
    array_new=[]
    for i in range(len(array)):
        array_new.append(array[i][n])
    minimym = min(array_new)
    index=array_new.index(minimym)
    return minimym, index

def index_max(array, n):
    array_new=[]
    for i in range(len(array)):
        array_new.append(array[i][n])
    maximym = max(array_new)
    indexmax=array_new.index(maximym)
    return maximym, indexmax


def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)



def del_face():
	q=0
	while q< len(faceList):
		dt=array_data[i][5]-faceList[q][5]
		if dt>2:
			del faceList[q]
		q+=1


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filefrom",type=str, default="",
	help="использование pi камеры")
ap.add_argument("-t", "--fileto", type=str, default="ID.txt",
	help="выбор файла")
ap.add_argument("-m", "--metod", type=str, default="w",
	help="выбор- открытие файла для чтения, записи, дозаписи")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


faceCount=0
array_finish_list=[]
array_data=[]

time_array = []
bad_face=1




try:
	with open("dictionary.pickle", "rb") as f:
		dictionary = []
		try:
			while True:
				dictionary.append(pickle.load(f))

		except (EOFError, pickle.UnpicklingError):
			pass

except FileNotFoundError:
	f = open('dictionary.pickle', 'wb')
	f.close
	dictionary = []

# чтение из файловой системы
os.chdir('info/')
for file in os.listdir('.'):
	if fnmatch.fnmatch(file, '*.jpg'):
		name=os.path.splitext(file)[0]
		name = float(name)
		time_array.append(name)

		
time_array = sorted(time_array, key=float)

os.chdir('..')


for n in range(len(time_array)):
    #
	
	frame= cv2.imread('info/'+str(time_array[n])+'.jpg')
	
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	height, width = frame.shape[:2]
	rects = detector(gray, 0)
	i=0
	if(len(rects) > 0):
		array_data_not_col=[]
		for rect in rects:
			(x, y, w, h) = face_utils.rect_to_bb(rect)

			if (x>0 and y>0 and x+w<width and y+h<height):
				i+=1
				start = time_array[n]

				shape_cam = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape_cam)

				gor = np.sqrt( (shape[0][0] - shape[16][0])**2 + (shape[0][1] - shape[16][1])**2 )
				vert = np.sqrt( (shape[8][0] - shape[27][0])**2 + (shape[8][1] - shape[27][1])**2 )
				otn=gor/vert

				face_descriptor= facerec.compute_face_descriptor(frame, shape_cam)
				
				facedata=["none",x, y, w, h, start, face_descriptor,False,0,0,"none",otn,False]
				array_data_not_col.append(copy.deepcopy(facedata))

		for m in range(len(array_data_not_col)):
			array_data_not_col[m][9]=int(i)
			print(array_data_not_col[m][9])
			array_data.append(copy.deepcopy(array_data_not_col[m]))

for l in range(len(array_data)):
	print(array_data[l][9])			

i=0
faceList = []

temporal_dist_array = []
array_for_apdata = []
array_namber = []

temporal_dist_array_3 = []
array_for_apdata_3 = []
array_namber_3 = []

while i < len(array_data):

#	for l in range(len(faceList)):
#		print(faceList[l][10])
#		print(faceList[l][9])
#		#print(array_data[i][5],array_data[i][10])
#	print(array_data[i][5])
#	del_face()
	
	for ll in range(len(faceList)):
		frame= cv2.imread('info/'+str(faceList[ll][5])+'.jpg')
		img=frame[int(faceList[ll][2]):int(faceList[ll][2])+int(faceList[ll][4]),int(faceList[ll][1]):int(faceList[ll][1])+int(faceList[ll][3])]
		cv2.imwrite('info2/'+str(faceList[ll][10])+'  '+str(i)+'.jpg',img)
#1
	del_face()
	if(len(faceList) == 0):
		
		for k in range(int(array_data[i][9])):
			array_data[i+k][10]=faceCount
			faceList.append(copy.deepcopy(array_data[i+k]))
			faceCount += 1


#2
	del_face()
	if(len(faceList) <= array_data[i][9]):
		
		#метки на обновление 
		metka=[False]*array_data[i][9]

		#каждому лицу из facelist выделить по столбцу из temporal_dist_array
		if len(temporal_dist_array)==0:
			for face in range(len(faceList)):
				temporal_dist_array.append([])
				array_for_apdata.append([])
				array_namber.append([])

		for f in range(len(faceList)):
			#созаем пустой массив для временной обработки
			rectangle_now=np.zeros((array_data[i][9],3))
			#перебор всех следующих элементов i-й и все лица что на этом кадре
			for rect in range(array_data[i][9]):
				x2=array_data[i+rect][1]
				y2=array_data[i+rect][2]

				x1=faceList[f][1]
				y1=faceList[f][2]
				dist_xy = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 )

				rectangle_now[rect][0]=x2
				rectangle_now[rect][1]=y2
				rectangle_now[rect][2]=dist_xy
			#поиск минимального расстояния
			minimym, index = index_min(rectangle_now, 2)
			#обновление массива
			metka[index]=True

#################################################################
			#dist=distance.euclidean(array_data[i+index][6], faceList[f][6])
			if len(temporal_dist_array[f])>0:
				dist=distance.euclidean(array_data[i+index][6], array_for_apdata[f][len(array_for_apdata[f])-1][6])
				#метка прерывания массива
				metka_all=False
				if dist<0.5:
					metka_all=True
					temporal_dist_array[f].append(dist)
					array_namber[f].append(i+index)
					array_for_apdata[f].append(copy.deepcopy(array_data[i+index]))
			else:
				dist=distance.euclidean(array_data[i+index][6], faceList[f][6])
				metka_all=False
				if dist>0.5:
					metka_all=True
					temporal_dist_array[f].append(dist)
					array_namber[f].append(i+index)
					array_for_apdata[f].append(copy.deepcopy(array_data[i+index]))


			#вносить полные изменения в array_data только в том случае если убедились что это был тот же человек
			if dist <0.5 and metka_all==False:
				faceList[f][1]=rectangle_now[index][0]
				faceList[f][2]=rectangle_now[index][1]
				array_data[i+index][10]=faceList[f][10] #запись обработаных данных в массив данных
				faceList[f][5]=array_data[i+index][5] # обновление времени
				#faceList[f][6]=array_data[i+index][6]
				

			#чтоб не было ошибок обновлять нужно постоянно только по координате и времени также нужно обновлять дескриптор чтобы адекватно работалафункция на добавление во временный массив
			faceList[f][1]=rectangle_now[index][0]
			faceList[f][2]=rectangle_now[index][1]
			faceList[f][5]=array_data[i+index][5]
			

			#metka_all==False указывает на других лиц в этих координатах не появились
			if len(temporal_dist_array[f])>0 and len(temporal_dist_array[f])<bad_face and metka_all==False:
				#запуск цикла на обновление номера в array_data
				for ni in range(len(temporal_dist_array[f])):
					faceList[len(faceList)-1][0]=copy.deepcopy(array_for_apdata[f][ni][0])
					faceList[len(faceList)-1][1]=copy.deepcopy(array_for_apdata[f][ni][1])
					array_data[array_namber[f][ni]][10]=faceList[f][10]
					faceList[f][5]=copy.deepcopy(array_for_apdata[f][ni][5])	
				del temporal_dist_array[f]
				del array_namber[f]
				del array_for_apdata[f]


		#выполнение удаленя и обновление данных если превысило 3


		n=0
		while n < len(temporal_dist_array):
			if len(temporal_dist_array[n])>=bad_face:
				del faceList[n]
				#добавляем новый faceList и по циклу выполняем присвоение номера в array_data
				faceList.append(copy.deepcopy(array_for_apdata[n][0]))
				faceList[len(faceList)-1][10]=faceCount

				for ni in range(len(temporal_dist_array[n])):
					faceList[len(faceList)-1][0]=copy.deepcopy(array_for_apdata[n][ni][0])
					faceList[len(faceList)-1][1]=copy.deepcopy(array_for_apdata[n][ni][1])
					array_data[array_namber[n][ni]][10]=faceCount
					faceList[len(faceList)-1][5]=copy.deepcopy(array_for_apdata[n][ni][5])
				del temporal_dist_array[n]
				del array_namber[n]
				del array_for_apdata[n]

				temporal_dist_array.append([])
				array_for_apdata.append([])
				array_namber.append([])

				faceCount+=1
			n+=1
############################################################
		for k in range(array_data[i][9]):
			if (metka[k]==False):
				array_data[i+k][10]=faceCount
				faceList.append(copy.deepcopy(array_data[i+k]))
				faceCount += 1

				temporal_dist_array.append([])
				array_for_apdata.append([])
				array_namber.append([])


#3
	del_face()
	if(len(faceList) > array_data[i][9]):

		#метка на удаление		
		metka=[0]*len(faceList)

		#каждому лицу из facelist выделить по столбцу из temporal_dist_array
		if len(temporal_dist_array_3)==0:
			for face in range(len(faceList)):
				temporal_dist_array_3.append([])
				array_for_apdata_3.append([])
				array_namber_3.append([])

		for rect in range(array_data[i][9]):
			rectangle_now=np.zeros((len(faceList),3))
			for f in range(len(faceList)):

				x2=array_data[i+f][1]
				y2=array_data[i+f][2]

				x1=faceList[rect][1]
				y1=faceList[rect][2]
				dist_xy = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 )

				rectangle_now[f][0]=x2
				rectangle_now[f][1]=y2
				rectangle_now[f][2]=dist_xy
			#поиск минимального расстояния
			minimym, index = index_min(rectangle_now, 2)
			metka[index]=True

#################################################################f-index  i+index - i+rect
			if len(temporal_dist_array_3[index])>0:
				dist=distance.euclidean(array_data[i+rect][6], array_for_apdata_3[index][len(array_for_apdata_3[index])-1][6])
				metka_all=False
				if dist<0.5:
					metka_all=True
					temporal_dist_array_3[index].append(dist)
					array_namber_3[index].append(i+index)
					array_for_apdata_3[index].append(copy.deepcopy(array_data[i+index]))
			else:
				dist=distance.euclidean(array_data[i+rect][6], faceList[index][6])
				metka_all=False
				if dist>0.5:
					metka_all=True
					temporal_dist_array_3[index].append(dist)
					array_namber_3[index].append(i+index)
					array_for_apdata_3[index].append(copy.deepcopy(array_data[i+index]))

			#вносить полные изменения в array_data только в том случае если убедились что это был тот же человек
			if dist <0.5 and metka_all==False:
				faceList[index][1]=rectangle_now[index][0]
				faceList[index][2]=rectangle_now[index][1]
				array_data[i+rect][10]=faceList[index][10] #запись обработаных данных в массив данных
				faceList[index][5]=array_data[i+rect][5] # обновление времени

			#чтоб не было ошибок обновлять нужно постоянно только по координате и времени
			faceList[index][1]=rectangle_now[index][0]
			faceList[index][2]=rectangle_now[index][1]
			faceList[index][5]=array_data[i+rect][5]

			#metka_all==False указывает на других лиц в этих координатах не появились
			if len(temporal_dist_array_3[index])<bad_face and metka_all==False:
				#запуск цикла на обновление номера в array_data
				for ni in range(len(temporal_dist_array_3[f])):
					faceList[index][0]=array_data[array_namber_3[ni]][0]
					faceList[index][1]=array_data[array_namber_3[ni]][1]
					array_data[array_namber_3[ni]][10]=faceList[index][10]
					faceList[index][5]=array_data[array_namber_3[ni]][5]
				temporal_dist_array_3[index]=[]

		#выполнение удаленя и обновление данных если превысило face_bad
		n=0
		while n < len(temporal_dist_array_3):
			if len(temporal_dist_array_3[n])>=bad_face:
				del faceList[n]
				#добавляем новый faceList и по циклу выполняем присвоение номера в array_data
				faceList.append(copy.deepcopy(array_for_apdata_3[n][0]))
				faceList[len(faceList)-1][10]=faceCount
				for ni in range(len(temporal_dist_array_3[n])):
					faceList[len(faceList)-1][0]=copy.deepcopy(array_for_apdata_3[n][ni][0])
					faceList[len(faceList)-1][1]=copy.deepcopy(array_for_apdata_3[n][ni][1])
					array_data[array_namber_3[n][ni]][10]=faceCount
					faceList[len(faceList)-1][5]=copy.deepcopy(array_for_apdata_3[n][ni][5])

				del temporal_dist_array_3[n]
				del array_for_apdata_3[n]	
				del array_namber_3[n]			
				
				temporal_dist_array_3.append([])
				array_for_apdata_3.append([])
				array_namber_3.append([])

				faceCount+=1
			n+=1

############################################################
		n=0
		while n < len(faceList):
			if(metka[n]==False):
				del faceList[n]
			n+=1

	i+=array_data[i][9]





#приствоение ID, выполняя сравнения со словарем
for i in range(len(dictionary)):
	print(dictionary[i][0])
	ID  = dictionary[i][0]
	for k in range(len(array_data)):
		namber_k=array_data[k][10]

		if (array_data[k][0]=="none"):

			n_array=[]
			array_namber_k=[]

			dist=distance.euclidean(dictionary[i][1], array_data[k][6])
			array_data[k][8]=dist
			list_1=copy.deepcopy(array_data[k])
			array_namber_k.append(list_1)
			namber=0
			n_array.append(namber)
			for n in range(len(array_data)-k-1):
				namber+=1
				if (array_data[k+n+1][10]==namber_k):

					n_array.append(namber)
					dist=distance.euclidean(dictionary[i][1], array_data[k+n+1][6])
					array_data[k+n+1][8]=dist
					list_1=copy.deepcopy(array_data[k+n+1])
					array_namber_k.append(list_1)

			minimym, index = index_min(array_namber_k, 8)

			if (minimym<=0.5):
				if array_namber_k[index][12]==False:
					for d in range(len(array_namber_k)):
						array_data[k+n_array[d]][0]=dictionary[i][0]
						array_data[k+n_array[d]][12]=True
						print('+++++++++++++++++++++++++++++++++++++++++++++++++++')





if len(dictionary)==0:
	ID=0
else:
	ID=dictionary[len(dictionary)-1][0]+1

#присвоение ID тем строкам которых нет в словаре
for i in range(len(array_data)):

	if (array_data[i][0]=="none"):
		array_data[i][0]=ID
		ID=ID+1

	for k in range(len(array_data)-i-1):

		namber_k=array_data[k+i+1][10]
		#if array_data[k+i+1][0]=="none":
			
		n_array=[]
		array_namber_k=[]

		dist=distance.euclidean(array_data[i][6], array_data[k+i+1][6])
		array_data[k+i+1][8]=dist
		list_1=copy.deepcopy(array_data[k+i+1])
		array_namber_k.append(list_1)
		namber=0
		n_array.append(namber)

		for n in range(len(array_data)-i-k-1):
			namber+=1
			if (array_data[k+i+n+1][10]==namber_k):
				n_array.append(namber)
				dist=distance.euclidean(array_data[i][6], array_data[k+i+n+1][6])
				array_data[i+k+n+1][8]=dist
				list_1=copy.deepcopy(array_data[k+i+n+1])
				array_namber_k.append(list_1)

		minimym, index = index_min(array_namber_k, 8)
		if (minimym<=0.5):
			#разрешить перезаписать ID если min меньше [8]
			if array_namber_k[index][12]==False:
				for d in range(len(array_namber_k)):
				
					array_data[k+i+n_array[d]][0]=array_data[i][0]
					array_data[k+i+n_array[d]][12]=True
					print('-------------------------------------------------')

#коррекция....
for i in range(len(array_data)):
	namber=array_data[i][10]
	ID=array_data[i][0]
	for k in range(len(array_data)-i-1):
		namber2=array_data[k+i+1][10]
		ID2=array_data[k+i+1][0]
		if ID!=ID2 and 	namber==namber2:
			array_data[k+i+1][0]=ID


#формировка строк для txt
for i in range(len(array_data)):
	#проверка на метку
	if (array_data[i][7]==False):
		#если строка не обработана, то переменной ID_2 присваиваем id этой строки
		ID_2=array_data[i][0]
		array_data[i][7]=True

		mass=[]
		mass.append(array_data[i])
		for k in range(len(array_data)-i-1):
			#проверка на метку и соответствие сторк i и i+k+1 на id
			if (array_data[k+i+1][7]==False) and (array_data[k+i+1][0]==ID_2):
				mass.append(array_data[k+i+1])
				if (mass[len(mass)-1][5]-mass[len(mass)-2][5]< 2):
					#ставим метку что строка обработана
					array_data[k+i+1][7]=True
				else:
					mass.pop()

		if len(mass)>0:
			array_f=mass
			finish_time=float(array_f[len(array_f)-1][5]+1)
			array_f[0][5]=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(array_f[0][5]))
			array_f[0][7]=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_time))
			maximym, indexmax = index_max(array_f, 11)
			array_f[0][6]=array_f[indexmax][6]
			array_finish_list.append(array_f[0])

d = open(args["fileto"], args["metod"])
for i in range(len(array_finish_list)):
	s= str(array_finish_list[i][0]) +"  "+ str(array_finish_list[i][5]) +"  " + str(array_finish_list[i][7])
	d.write(s+ '\n')
d.close()


#цикл для создания словаря 
array_dictionary_important=[]
metka=[False]*len(array_finish_list)

for i in  range(len(array_finish_list)):
	if metka[i]==False:
		ID=array_finish_list[i][0]

		temporal_array=[]
		temporal_array.append(array_finish_list[i])
		metka[i] = True
		for k in range(len(array_finish_list)-i-1):
			if (metka[k+i+1]==False) and (array_finish_list[k+i+1][0]==ID):
				metka[k+i+1] = True
				temporal_array.append(array_finish_list[k+i+1])
		maximym, indexmax = index_max(temporal_array, 11)

for i in range(len(array_data)):
	important= [array_data[i][0], array_data[i][6], array_data[i][11]]
	array_dictionary_important.append(important)

#формирование обновленного словаря
for i in range(len(array_dictionary_important)):
	dictionary.append(array_dictionary_important[i])


dictionary_updata=[]
metka=[False]*len(dictionary)

#групировка по одному ID и сортирвка нужных дескрипторов
for i in range(len(dictionary)):
	if metka[i]==False:
		ID=dictionary[i][0]

		temporal_array=[]
		temporal_array.append(dictionary[i])
		metka[i] = True
		for k in range(len(dictionary)-i-1):
			if (metka[k+i+1]==False) and (dictionary[k+i+1][0]==ID):
				metka[k+i+1] = True
				temporal_array.append(dictionary[k+i+1])
		maximym, indexmax = index_max(temporal_array, 2)
		minimym, indexmin = index_min(temporal_array, 2)
		updata_max= temporal_array[indexmax]
		updata_min= temporal_array[indexmin]
		dictionary_updata.append(updata_max)
		dictionary_updata.append(updata_min)
		
		all_deskriptor=[]
		for d in range(len(temporal_array)):
		
			for d1 in range(len(temporal_array)-d-1):
				dist=distance.euclidean(temporal_array[d][1], temporal_array[d1+d+1][1])
				for_all_deskriptor=[ID,temporal_array[d][1],temporal_array[d1+d+1][1],dist,temporal_array[d][2],temporal_array[d1+d+1][2]]
				all_deskriptor.append(for_all_deskriptor)
		#сортировка и выбор максимальных
		all_deskriptor.sort(key=lambda i: i[3])
		x=0
		for l in reversed(all_deskriptor):
			if x<5:
				
				deskriptor1=[l[0],l[1],l[3]]
				deskriptor2=[l[0],l[2],l[4]]
				dictionary_updata.append(deskriptor1)
				dictionary_updata.append(deskriptor2)
			x+=1
#цикл на удаление совпадающих дескрипторов
i=0
while i < len(dictionary_updata):
	ID=dictionary_updata[i][0]
	k=0
	while k < len(dictionary_updata)-i-1:
		dist=distance.euclidean(dictionary_updata[i][1], dictionary_updata[i+k+1][1])
		if dictionary_updata[i+k+1][0]==ID and dist==0:
			del dictionary_updata[i+k+1]

		k+=1
	i+=1
with open("dictionary.pickle","wb") as f:
	for i in range(len(dictionary_updata)):
		pickle.dump(dictionary_updata[i], f)

d = open("test.txt", "w")
for i in range(len(array_data)):
	s=str(array_data[i][0])+'   '+str(array_data[i][5])+'   '+ str(array_data[i][10])
	d.write(s+ '\n')
	d.write('\n')
d.close()


d = open("dictionary.txt", "w")
for i in range(len(dictionary_updata)):
	s=str(dictionary_updata[i])
	d.write(s+ '\n')
	d.write('\n')
d.close()
