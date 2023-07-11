import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import random
#use glob to get all the csv files
# in the folder
path = os.getcwd()
csv_files = glob.glob(os.path.join('D:\kaggle\predicted data', "*.csv"))
r = random.random()
b = random.random()
g = random.random()
color = (r, g, b)
print("\n\nThe Housing price prediction with different constrains or features has been processed for statical analysis is shown below ")
print("\n\n")
i=1
# loop over the list of csv files
for f in csv_files:
    # read the csv file
    df = pd.read_csv(f)
    print("The House price prediction : prediction no",i)
    df.set_index("Id").plot(c=color)
    plt.title('Random forest Prediction')
    plt.grid()
    plt.legend()
    plt.show()
    i+=1
path = os.getcwd()
csv_file = glob.glob(os.path.join('D:\kaggle\Statical analysis data of prediction', "*.csv"))
print(" The over all predictions are classified and processed into Statical format is shown below:\n\n\n ")
for e in csv_file:
    df = pd.read_csv(e)
    df.set_index("Id").plot()
    plt.title('Over All Analysis ')
    plt.grid()
    plt.legend()
    plt.show()st
