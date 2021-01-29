from shutil import ignore_patterns
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import numpy as np

class SimpleDrawer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.drawMeanTicket()
        self.drawNrOrders()
        self.drawProfit()
        self.drawTBO()
        self.drawTTS()
        self.drawChurn()

        self.drawHistogram(self.dataset.getDataByName("utils_tbo"), title="Histogram TBO", pic_name="")
        self.drawHistogram(self.dataset.getDataByName("utils_tts"), title="Histogram TTS", pic_name="")
        self.drawHistogram(self.dataset.getDataByName("customers_tbo"), title="Histogram TBO-Cliente", pic_name="")
        self.drawHistogram(self.dataset.getDataByName("orders_total_price"), title="Histogram Ticket", pic_name="")
        self.drawHistogram(self.dataset.getDataByName("customers_avg_order"), title="Average Order per Customer", pic_name="")

        self.drawTopProducts()
        self.draw_category()
        self.pearsonCorrelation()

    def draw_category(self):
        product = self.dataset.getDataByName("order_details_product_id")
        p_prod_id = self.dataset.getDataByName("products_product_id")
        p_cat_id = self.dataset.getDataByName("products_category_id")

        c_cat_id = self.dataset.getDataByName("categories_category_id")
        c_cat_name = self.dataset.getDataByName("categories_category_name")
        
        cat_id_res = []
        for i in product:
            cat_id = p_cat_id[np.where(p_prod_id == i)[0]]
            cat_id_res.append(cat_id)
        cat_id_res = np.array(cat_id_res).reshape([-1,1])

        quantity = self.dataset.getDataByName("order_details_quantity")
        u_price = self.dataset.getDataByName("order_details_unit_price")
        discount = self.dataset.getDataByName("order_details_discount")

        raw_values = u_price * quantity * (1 - discount)
        product_quantity = pd.DataFrame(np.concatenate([cat_id_res, quantity], axis=1)).groupby(0, as_index=False).sum().to_numpy()
        product_value = pd.DataFrame(np.concatenate([cat_id_res, raw_values], axis=1)).groupby(0, as_index=False).sum().to_numpy()

        label_q = []
        label_v = []
        for i in range(product_quantity.shape[0]):
            label_q.append(c_cat_name[np.where(c_cat_id == product_quantity[i][0])[0]])
            label_v.append(c_cat_name[np.where(c_cat_id == product_value[i][0])[0]])
        label_q = np.array(label_q).reshape(-1)
        label_v = np.array(label_v).reshape(-1)
        self.drawPie(label_q, product_quantity[:, 1])
        self.drawPie(label_v, product_value[:, 1])


    def drawTopProducts(self):
        product = self.dataset.getDataByName("order_details_product_id")
        quantity = self.dataset.getDataByName("order_details_quantity")
        u_price = self.dataset.getDataByName("order_details_unit_price")
        discount = self.dataset.getDataByName("order_details_discount")

        raw_values = u_price * quantity * (1 - discount)
        product_quantity = pd.DataFrame(np.concatenate([product, quantity], axis=1)).groupby(0, as_index=False).sum().to_numpy()
        product_value = pd.DataFrame(np.concatenate([product, raw_values], axis=1)).groupby(0, as_index=False).sum().to_numpy()
        
        sorter = np.argsort(product_quantity[:,1])
        product_quantity = product_quantity[np.flip(sorter)]
        sorter = np.argsort(product_value[:,1])
        product_value = product_value[np.flip(sorter)]


        p_prod_id = self.dataset.getDataByName("products_product_id")
        names = self.dataset.getDataByName("products_product_name")

        ###draw only the 20 most used products, and top 20 products in values sold
        quantities = product_quantity[:20,1]
        values = product_value[:20,1]
        
        labels_q = []
        labels_v = []
        for i in range(20):
            index = np.where(p_prod_id == product_quantity[i][0])
            labels_q.append(names[index][0])
            index = np.where(p_prod_id == product_value[i][0])
            labels_v.append(names[index][0])

        self.drawColumn(labels_q,quantities)
        self.drawColumn(labels_v,values)
        
    
    def drawMeanTicket(self):
        orderdate = self.dataset.getDataByName("orders_order_date")
        val_per_order = self.dataset.getDataByName("orders_total_price") 
       
        month_of_orders = self.dateToCount(self.ignoreDays(orderdate))
        val_per_order = pd.DataFrame(np.concatenate([month_of_orders, val_per_order], axis=1)).groupby(0).mean().to_numpy() #mean ticket per month
        print(np.mean(val_per_order))        
        date_to_plot = np.unique(self.ignoreDays(orderdate))
        self.drawLine(date_to_plot[1:-2], val_per_order[1:-2], labelY="Order($)", title="Mean Ticket over Time", pic_name="mean_ticket")  ##[1:-2] eliminate first and last values as they represent incomplete months

    def drawNrOrders(self): 
        orderdate = self.ignoreDays(self.dataset.getDataByName("orders_order_date"))
        y = np.bincount(self.dateToCount(orderdate.reshape(-1)))
        self.drawLine(np.unique(orderdate)[1:-2], y[1:-2], labelY="Nr. Orders", title="Number of Orders", pic_name="nr_orders")  ##[1:-2] eliminate first and last values as they represent incomplete months
        
    def drawProfit(self):
        orderdate = self.dataset.getDataByName("orders_order_date")
        freight = self.dataset.getDataByName("orders_freight")

        val_per_order = self.dataset.getDataByName("orders_total_price") #value summed for each order
        profit = val_per_order - freight    #### product cost not available
        month_of_orders = self.dateToCount(self.ignoreDays(orderdate))
        profit = pd.DataFrame(np.concatenate([month_of_orders, profit], axis=1)).groupby(0).sum().to_numpy() #total profit per month
        months_to_plot = np.unique(self.ignoreDays(orderdate))
        self.drawLine(months_to_plot[1:-2], profit[1:-2], labelY="Profit", title="Profit (without product cost)", pic_name="profit") ##[1:-2] eliminate first and last values as they represent incomplete months

    def drawTBO(self):
        tbo = self.dataset.getDataByName("utils_tbo")
        print(np.mean(tbo))
        tbo_date = self.dataset.getDataByName("utils_tbo_dates")
        tbo_month = self.ignoreDays(tbo_date)
        tbo = pd.DataFrame(np.concatenate([self.dateToCount(tbo_month), tbo], axis=1)).groupby(0).mean().to_numpy()
        self.drawLine(np.unique(tbo_month), tbo, labelY="time between orders (days)", title="Mean Time between Orders", pic_name="tbo_time")  ##[1:-2] eliminate first and last values as they represent incomplete months
        
    def drawTTS(self):
        abs_time = self.dataset.getDataByName("utils_tts")
        relative_time = self.dataset.getDataByName("utils_tts_relative")
        month = self.ignoreDays(self.dataset.getDataByName("utils_tts_dates"))
        abs_time = pd.DataFrame(np.concatenate([self.dateToCount(month), abs_time], axis=1)).groupby(0).mean().to_numpy()
        relative_time = pd.DataFrame(np.concatenate([self.dateToCount(month), relative_time], axis=1)).groupby(0).mean().to_numpy()
        y = np.concatenate([abs_time, relative_time], axis=1)
        self.drawLine(np.unique(month)[1: -2], y[1: -2], labelY="time to Ship", title="Mean Time to Ship Orders(absolute and Relative)", pic_name="ship_time", labels=["TTS(days)", "TTS / total time (%)"])  ##[1:-2] eliminate first and last values as they represent incomplete months
        
    def drawChurn(self):
        churn = self.ignoreDays(self.dataset.getDataByName("customers_churn"))
        first = self.ignoreDays(self.dataset.getDataByName("customers_first_order"))
        month = np.unique(self.ignoreDays(self.dataset.getDataByName("orders_order_date")))[1:-2]
        freq = np.empty([month.shape[0], 2])
        for i in range(month.shape[0]):
            freq[i, 0] = np.bincount(churn!=month[i])[0]
            freq[i, 1] = np.bincount(first!=month[i])[0]
        
        delay = self.dataset.getDataByName("utils_tts_delay")
        del_month = self.ignoreDays(self.dataset.getDataByName("utils_tts_dates"))
        delay = pd.DataFrame(np.concatenate([self.dateToCount(del_month), delay], axis=1)).groupby(0).sum().to_numpy()
        freq = np.concatenate([freq, delay[1:-2]], axis=1)
        
        self.drawLine(month, freq, labelY="Customers", title="Churns According to buying frequency", pic_name="churn_time", labels=["Churn", "New Customers", "Delays to Send"])

    
    def pearsonCorrelation(self):
        churn = self.dataset.getDataByName("customers_churn_rate").reshape(-1)
        ticket = self.dataset.getDataByName("customers_avg_order").reshape(-1)
        delta_ticket = self.dataset.getDataByName("customers_delta_order").reshape(-1)
        tts = self.dataset.getDataByName("customers_tts").reshape(-1)
        tbo = self.dataset.getDataByName("customers_tbo").reshape(-1)
        delay = self.dataset.getDataByName("customers_delay_rate").reshape(-1)

        corrs = np.round(np.corrcoef([churn, ticket, delta_ticket, tts, tbo, delay]), decimals=2)
        
        labels = ["churn rate", "average Purchase Value", "delta Purchase Value", "Time to Send", "Time between orders", "delay rate"]

        fig, ax = plt.subplots()
        im = ax.imshow(corrs)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, corrs[i, j],
                            ha="center", va="center", color="w")

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set_title("Pearson Correlation Between Customer variables")
        fig.tight_layout()
        plt.show()
    
    def drawHistogram(self, array, labelX="interval", labelY="frequency", title="", pic_name=""):
        print(np.mean(array))
        y, x = np.histogram(array, bins=20)
        print(y)
        print(np.sum(y))
        plt.figure(figsize=(10,6))
        plt.hist(x[:-1], x, weights=y)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        if(pic_name != ""):
            plt.savefig(path.join("./charts", pic_name + ".png"))
        plt.show()


    def drawLine(self, x, y, labelX="Time", labelY="value", title="", pic_name="", labels=None):
        plt.figure(figsize=(10,6))
        if(labels == None):
            plt.plot(x, y)
        else:
            for i in range(len(labels)):
                plt.plot(x, y[:,i], label=labels[i])
            plt.legend(loc="upper right")
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        if(pic_name != ""):
            plt.savefig(path.join("./charts", pic_name + ".png"))
        plt.show()

    def drawColumn(self, labels, values, labelX="interval", labelY="frequency", title="", pic_name=""):
        y_pos = np.arange(len(labels))

        plt.bar(y_pos, values, align='center', alpha=0.5)
        plt.xticks(y_pos, labels, rotation=90)
        plt.ylabel(labelY)
        plt.title(title)
        plt.gcf().subplots_adjust(bottom=0.5)
        if(pic_name != ""):
            plt.savefig(path.join("./charts", pic_name + ".png"))
        plt.show()

    def drawPie(self, labels, sizes):

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

    def ignoreDays(self, array):
        x = np.datetime_as_string(array)
        def vect_cut(x):
            return x[:7]
        func = np.vectorize(vect_cut)

        return (func(x)).astype('datetime64')
    
    def dateToCount(*args):
        array = args[1]
        if(len(args)==2):
            reference = args[1][0]
        else:
            reference = args[2]
        x = array - reference
        return x.astype('int32')


