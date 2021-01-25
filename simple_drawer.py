from shutil import ignore_patterns
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import numpy as np

class SimpleDrawer:
    def __init__(self, dataset):
        self.dataset = dataset
        #self.drawMeanTicket()
        #self.drawNrOrders()
        #self.drawProfit()
    
    def drawMeanTicket(self):
        orderdate = self.dataset.getDataByName("orders_order_date")
        val_per_order = self.dataset.getDataByName("orders_total_price") 
       
        month_of_orders = self.monthToInt(self.ignoreDays(orderdate))
        val_per_order = pd.DataFrame(np.concatenate([month_of_orders, val_per_order], axis=1)).groupby(0).mean().to_numpy() #mean ticket per month
        date_to_plot = np.unique(self.ignoreDays(orderdate))
        self.drawLine(date_to_plot[1:-2], val_per_order[1:-2], labelY="Order($)", title="Mean Ticket over Time", pic_name="")  ##[1:-2] eliminate first and last values as they represent incomplete months

    def drawNrOrders(self): 
        orderdate = self.ignoreDays(self.dataset.getDataByName("orders_order_date"))
        y = np.bincount(self.monthToInt(orderdate.reshape(-1)))
        self.drawLine(np.unique(orderdate)[1:-2], y[1:-2], labelY="Nr. Orders", title="Number of Orders", pic_name="nr_orders")  ##[1:-2] eliminate first and last values as they represent incomplete months
        
    def drawProfit(self):
        orderdate = self.dataset.getDataByName("orders_order_date")
        freight = self.dataset.getDataByName("orders_freight")

        val_per_order = self.dataset.getDataByName("orders_total_price") #value summed for each order
        profit = val_per_order - freight    #### product cost not available
        month_of_orders = self.monthToInt(self.ignoreDays(orderdate))
        profit = pd.DataFrame(np.concatenate([month_of_orders, profit], axis=1)).groupby(0).sum().to_numpy() #total profit per month
        months_to_plot = np.unique(self.ignoreDays(orderdate))
        self.drawLine(months_to_plot[1:-2], profit[1:-2], labelY="Profit", title="Profit (without product cost)", pic_name="profit") ##[1:-2] eliminate first and last values as they represent incomplete months

    def drawLine(self, x, y, labelX="Time", labelY="value", title="", pic_name=""):
        plt.figure(figsize=(10,6))
        plt.plot(x, y)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        if(pic_name != ""):
            plt.savefig(path.join("./charts", pic_name + ".png"))
        plt.show()

    def ignoreDays(self, array):
        x = np.datetime_as_string(array)
        def vect_cut(x):
            return x[:7]
        func = np.vectorize(vect_cut)

        return (func(x)).astype('datetime64')
    
    def monthToInt(self, array):
        x = array - np.min(array)
        return x.astype('int32')



