import os
path = "./SegmentationClassAug" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
s = []


with open("./train.txt",'w+') as file:
	for file_name in files:
		file.write("/JPEGImages/" + file_name[:-4] + ".jpg" + " " + "/SegmentationClassAug/" + file_name)
		file.write('\n')

