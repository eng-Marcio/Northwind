from os import listdir
from os import path
import numpy as np
from numpy.lib.index_tricks import CClass
import pandas as pd


class DBReader:
    def __init__(self):
        self.dataSet = []
        self.readCSVs()
        self.createAdditionals()
        

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


    def createAdditionals(self):

        ### add total order value to each order
        orderid = self.getDataByName("order_details_order_id")
        unit_price = self.getDataByName("order_details_unit_price")
        quantity = self.getDataByName("order_details_quantity")
        discount = self.getDataByName("order_details_discount")

        values = unit_price * quantity * ( 1 - discount) # value productwise of the order
        val_per_order = pd.DataFrame(np.concatenate([orderid, values], axis=1)).groupby(0).sum().to_numpy() #value summed for each order
        self.dataSet.append(["orders_total_price", val_per_order])

        ###TBO: time between orders: the interval through which the customer makes an order
        order_date = self.toNumber(self.getDataByName("orders_order_date"))
        order_customer = self.getDataByName("orders_customer_id")

        sorter = np.argsort(order_customer.reshape(-1), kind='stable')
        order_date = order_date[sorter].reshape(-1)
        order_customer = order_customer[sorter].reshape(-1)
        
        tbo_all = []
        tbo_all_dates = []
        checked_customers = []
        tbo_customer = []

        target = order_customer[0]
        date_buffer = []
        t0 = np.min(self.getDataByName("orders_order_date"))

        for i in zip(order_customer, order_date):
            if(i[0] == target):
                date_buffer.append(i[1])
                if(len(date_buffer) > 1):
                    tbo_all.append(date_buffer[-1] - date_buffer[-2])
                    tbo_all_dates.append(t0 + np.timedelta64(i[1], 'D'))
            else:
                checked_customers.append(target)
                tbo_customer.append((date_buffer[-1] - date_buffer[0])/len(date_buffer)) ##remember: customers with 1 order will have TBO zero
                date_buffer = [i[1]]
                target = i[0]
        
        tbo_all_dates =  np.array(tbo_all_dates)
        tbo_all =  np.array(tbo_all)
        sorter = np.argsort(tbo_all_dates)
        tbo_all_dates = tbo_all_dates[sorter]
        tbo_all = tbo_all[sorter]
        
        self.dataSet.append(["utils_tbo_dates", tbo_all_dates])
        self.dataSet.append(["utils_tbo", tbo_all])
        
        #create a tbo collumn for cutomer table
        c_c_id = self.getDataByName("customers_customer_id")
        tab = np.empty(c_c_id.shape)
        for i in range(len(checked_customers)):
            tab[np.where(c_c_id == checked_customers[i])] = tbo_customer[i]
        self.dataSet.append(["customers_tbo", tab])



        
    def toNumber(self, array):
        x = array - np.min(array)
        return x.astype('int32')

        


        
            
            
