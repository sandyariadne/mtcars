#import module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Import CSV mtcars
data = pd.read_csv('mtcars.csv')
data.head()

# kolom yang ada
data.columns
# keterangan variabel(feature):
# [1]	mpg   = Miles/(US) gallon
# [2]	cyl   = Number of cylinders
# [3]	disp  = Displacement (cu.in.)
# [4]	hp    = Gross horsepower
# [5]	drat  = Rear axle ratio
# [6]	wt    = Weight (1000 lbs)
# [7]	qsec  = 1/4 mile time
# [8]	vs    = Engine (0 = V-shaped, 1 = straight)
# [9]	am    = Transmission (0 = automatic, 1 = manual)
# [10]	gear  = Number of forward gears
# [11]	carb  = Number of carburetors

# menghilangkan NaN
data = data.dropna()


# membuat kolom baru bernama mpg_level
#Jika mpg < 20 maka mpg_level = low
#Jika mpg berkisar 20-30 maka mpg_level = medium
#Jika mpg > 30 maka mpg_level = hard

def mpglevel(mpg):
    if mpg < 20:
        return("low")
    elif mpg > 30:
        return("hard")
    else:
        return("medium")
        
data['mpg_level'] = data.apply(lambda x: mpglevel(x['mpg']),axis=1)

# summary
data.describe()

# visualisasi data
# distribusi mpg_level
sns.set(rc={'figure.figsize':(7,7)})
sns.countplot(x="mpg_level", data=data)

# distribusi tiap variabel
data.hist(['mpg', 'disp', 'hp', 'drat', 'qsec', 'carb'],
       figsize=(10,10))
plt.show()

# barplot transmisi berdasarkan mesin
print("vs= Engine (0 = V-shaped, 1 = straight)")
print("am= Transmission (0 = automatic, 1 = manual)")
print("Number of models: "+str(len(data.index)))
sns.countplot(x="am", hue="vs", data=data)


#tabel korelasi tiap feature
correlation = data.corr()
plt.figure(figsize = (10, 10))
sns.heatmap(correlation, vmax = 1, square = True, annot = True)

# data frame baru 
numcols = ['mpg', 'disp', 'hp', 'drat', 'wt', 'qsec']
catcols = ['mpg_level']
df = data[numcols+catcols]
# plot mpg_level berdasarkan Displacement (cu.in.), Gross horsepower, Rear axle ratio, Weight (1000 lbs), and 1/4 mile time
sns.pairplot(df,hue="mpg_level")
