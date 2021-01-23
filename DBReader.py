from os import listdir
from os import path
import numpy as np


class DBReader:
    def __init__(self):
        self.dataSet = []
        self.readCSVs()
        

    def readCSVs(self):
        root_dir = "./data"
        for f in listdir(root_dir):
            file_table = np.genfromtxt(path.join(root_dir, f), delimiter=';', dtype="str", comments=None)
            header = np.char.add(f.split('.')[0] + "_", file_table[0])
            content = []
            file_table = file_table[1:]
            for i in range(file_table.shape[1]):
                try:
                    data = np.copy(file_table[:,i:i+1])
                    data[data==''] = 0
                    data = data.astype("float")
                except ValueError:
                    try:
                        data = np.copy(file_table[:,i:i+1]).astype(np.datetime64)
                    except ValueError:
                        data = np.copy(file_table[:,i:i+1])
                content.append(data)
        
            for pair in zip(header, content):
                self.dataSet.append(pair)

    def getDataByName(self, name):
        for i in self.dataSet:
            if(i[0] == name):
                return i[1]

    def appendList(self, data):
        self.dataSet.append(data)


        
            
            
