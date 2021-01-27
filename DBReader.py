from os import listdir
from os import path
import numpy as np
from numpy.core.fromnumeric import argsort
from numpy.lib.index_tricks import CClass
import pandas as pd


class DBReader:
    def __init__(self):
        self.dataSet = []
        self.readCSVs()
        self.createAdditionals()
        self.calculateCustomerData()
        self.calculateChurn()
        

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
        order_date = self.dateToCount(self.getDataByName("orders_order_date"))
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
        
        self.dataSet.append(["utils_tbo_dates", tbo_all_dates.reshape([-1,1])])
        self.dataSet.append(["utils_tbo", tbo_all.reshape([-1,1])])
        
        ### calculate and add TTS: time to ship the orders given from the customer
        shipdate = self.getDataByName("orders_shipped_date")
        orderdate = self.getDataByName("orders_order_date")
        requireddate = self.getDataByName("orders_required_date")
        order0 = orderdate[0][0]
        shipdate = self.dateToCount(shipdate, order0)
        orderdate = self.dateToCount(orderdate, order0)
        requireddate = self.dateToCount(requireddate, order0)
        orders_delayed = (shipdate.reshape(-1)>requireddate.reshape(-1)).reshape([-1,1])
        
        #delete where products were not shipped yet
        valids = np.where(shipdate > 0)
        shipdate = shipdate[valids].reshape([-1,1])
        orderdate = orderdate[valids].reshape([-1,1])
        requireddate = requireddate[valids].reshape([-1,1])
        date_set = self.getDataByName("orders_order_date")[valids]
        orders_delayed = orders_delayed[valids]

        abs_time = shipdate - orderdate
        relative_time = np.divide((shipdate - orderdate),(requireddate - orderdate))*100 #percentage of delivery time took to send the product
        self.dataSet.append(["utils_tts_dates", date_set.reshape([-1,1])])
        self.dataSet.append(["utils_tts", abs_time])
        self.dataSet.append(["utils_tts_relative", relative_time])
        self.dataSet.append(["utils_tts_delay", orders_delayed.reshape([-1,1])])


    def calculateCustomerData(self):
        first_date = self.getDataByName("orders_order_date")[0][0]

        order_customer = self.getDataByName("orders_customer_id").reshape(-1)
        order_prices = self.getDataByName("orders_total_price").reshape(-1)
        order_dates = self.dateToCount(self.getDataByName("orders_order_date").reshape(-1), first_date)
        order_ship_date = self.dateToCount(self.getDataByName("orders_shipped_date").reshape(-1), first_date)
        order_required_date = self.dateToCount(self.getDataByName("orders_required_date").reshape(-1), first_date)
        customer_id = self.getDataByName("customers_customer_id").reshape(-1)
        
        
        customer_average_ticket = np.empty(customer_id.shape[0])
        customer_delta_ticket = np.empty(customer_id.shape[0])
        customer_tbo = np.empty(customer_id.shape[0])
        customer_delta_tbo = np.empty(customer_id.shape[0])
        customer_delay_rate = np.empty(customer_id.shape[0])
        customer_tts = np.empty(customer_id.shape[0])
        
        customer_churn = np.empty(customer_id.shape[0])
        last_day = order_dates[-1]

        for i in range(customer_id.shape[0]):
            customer_orders = np.where(order_customer == customer_id[i])
            dates = order_dates[customer_orders]
            if(dates.shape[0] < 2):
                customer_average_ticket[i] = 0
                customer_delta_ticket[i] = 0
                customer_tbo[i] = 0
                customer_delta_tbo[i] = 0
                customer_delay_rate[i] = 0
                customer_tts[i] = 0
                customer_churn[i] = 1
                continue
            ship_date = order_ship_date[customer_orders]
            required_date = order_required_date[customer_orders]
            intervals = np.diff(dates)
            values = order_prices[customer_orders]

            tbo = np.mean(intervals)
            delta_tbo = np.mean(np.diff(intervals))
            c_ticket = np.mean(values)
            c_delta_ticket = np.mean(np.diff(values))
            delays = np.sum(ship_date > required_date)
            
            shipped = np.where(ship_date > 0)
            tts = np.mean(ship_date[shipped] - dates[shipped])

            if (intervals.shape[0] == 0):
                churn_rate = 1
            else:
                churn_rate = (intervals[-1] + last_day - dates[-1]) / (2* tbo + np.finfo(float).eps)
            
            customer_average_ticket[i] = c_ticket
            customer_delta_ticket[i] = c_delta_ticket
            customer_tbo[i] = tbo
            customer_delta_tbo[i] = delta_tbo
            customer_delay_rate[i] = delays / (float(dates.shape[0]) + np.finfo(float).eps)
            customer_tts[i] = tts
            customer_churn[i] = churn_rate

        self.dataSet.append(["customers_avg_order", customer_average_ticket])
        self.dataSet.append(["customers_delta_order", customer_delta_ticket])
        self.dataSet.append(["customers_tbo", customer_tbo])
        self.dataSet.append(["customers_delta_tbo", customer_delta_tbo])
        self.dataSet.append(["customers_delay_rate", customer_delay_rate])
        self.dataSet.append(["customers_tts", customer_tts])
        self.dataSet.append(["customers_churn_rate", customer_churn])
        
            




        

    

    def calculateChurn(self):
        customer_id = self.getDataByName("customers_customer_id")
        tbo_costumer = self.getDataByName("customers_tbo")
        orders_ct = self.getDataByName("orders_customer_id")
        order_date = self.dateToCount(self.getDataByName("orders_order_date")).reshape(-1)
        
        last_orders = np.empty(customer_id.shape[0])
        first_orders = np.full(customer_id.shape[0], -100)
        #get 2 last orders of each customers
        for i in range(order_date.shape[0]):
            index = np.where(customer_id == orders_ct[i])[0]
            if(first_orders[index] == -100):
                first_orders[index] = order_date[i]
            last_orders[index] = order_date[i]
        
        #condition created to determine churn: if last order + 1.5*TBO < current date
        last_date = order_date[-1]
        churn, = np.where((last_orders + (1.5 * tbo_costumer.reshape(-1))) < last_date)
        start_date = self.getDataByName("orders_order_date")[0][0]
        churn_date = np.full(customer_id.shape[0], 0, dtype='datetime64[D]')
        firts_date = np.full(customer_id.shape[0], 0, dtype='datetime64[D]')
        for i in churn:
            churn_date[i] = start_date + np.timedelta64(int(last_orders[i]), 'D')
        for i in range(firts_date.shape[0]):
            firts_date[i] = start_date + np.timedelta64(int(first_orders[i]), 'D')
        
        self.dataSet.append(["customers_churn", churn_date])
        self.dataSet.append(["customers_first_order", firts_date])


    def dateToCount(*args):
        array = args[1]
        if(len(args)==2):
            reference = args[1][0]
        else:
            reference = args[2]
        x = array - reference
        return x.astype('int32')

        


        
            
            
