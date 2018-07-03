import os
from shutil import copy
from sys import argv

#print(sys.argv)

SOURCE_DIR = "./" + argv[1]
DESTINATION_DIR = "./data"

listdir = os.listdir(SOURCE_DIR)
#print(listdir)
for i in listdir:
	filename = i.split('-')
	if(filename[0] == 'note'):
		outDir = filename[0]+'-'+filename[1]+'-'+filename[2]
	elif(filename[0] == 'rest'):
		outDir = filename[0]+'-'+filename[1]
	#print(outDir)
	if not os.path.exists(DESTINATION_DIR+"/"+outDir):
		os.makedirs(DESTINATION_DIR+"/"+outDir)
	print('copying '+i+' => '+DESTINATION_DIR+"/"+outDir)
	copy(SOURCE_DIR+'/'+i, DESTINATION_DIR+"/"+outDir)
