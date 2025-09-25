import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error  
import math
# This was the last part that I imported now I don't really get it so I be flexing on the knowledge later .
import pylab #(Like what the fuck does pylab do?)
import scipy.stats as stats




path = kagglehub.dataset_download("kolawale/focusing-on-mobile-app-or-website")

print("Dataset path:", path)

# # #List all files in the dataset folder
files = os.listdir(path)
print("Files in dataset:", files)


file_name = files[0] 
file_path = os.path.join (path, file_name)

project_folder = r"C:\Users\precious\OneDrive\Desktop\Machine learning project"
shutil.copy( file_path, project_folder)
print(f"Copied {file_name} to project folder.")
df = pd.read_csv(file_path)
print(df.head())
df.info()
print (df.describe())

#EDA Exploritary Data Analysis  

sns.jointplot( x= "Time on Website", y = "Yearly Amount Spent", data = df, alpha = 0.5)
plt.show()


sns.jointplot(x= "Time on App", y = "Yearly Amount Spent", data = df, alpha = 0.5)
plt.show()

plt.close ()

sns.pairplot (df , kind = 'scatter' , plot_kws={'alpha': 0.4}  )
plt.show ()
 
sns.lmplot (x = 'Length of Membership',
            y = 'Yearly Amount Spent',
            data = df,
            scatter_kws ={'alpha' :0.4})

plt.show()
            
x = df [['Avg. Session Length' , 'Time on App', 'Time on Website', 'Length of Membership']]            
y = df ['Yearly Amount Spent']
x_train, x_test, y_train, y_test = train_test_split ( x, y, test_size = 0.3, random_state= 42)           #so basically I don't really have a full fledge understanding of this shit
#  but by now we know that we will usually train on more data than the amount of data used to test. Here the test size was dividied by 100 
#obviously, but I do not know why though let's go on from there!!!, in the aspect of 42, I will just say study indexing or whatever becuase 
# when I did it was if you want to print a certain number consecutively, just do it .


print (x_train)
print (x_test) #I will be leaving this comment so when I get back, I will know what exaclty confused me here 
#I have no idea, how the 350 rows x 4 columns etc work but the programmer said it was training data how? 
# I need more dept on it, so by the time I come back here, i want to know how, the idea behind it etc 
print (y_train)
print (y_test)  # I will be adding them as comment for what it's worth 


lm = LinearRegression ()

lm.fit(x_train , y_train) # so from my shallow understanding this is telling the linear regression to take the data samples 
#from the X like training with the X, you don't have to manaully do anything, the Linear regression model will do it for you
#so it's just giving it the data and specify, and it was also stated that what was trained from 'x' should be added to 'y' I don't 
# know why. But with future practice, I will get it done.
   
print (LinearRegression)

print (lm.coef_  ) # so basically, this print's the coeficeint, now I don't know what the coeficient means but it does have 
# something to do with the formula y= b+ bx stuff like that.
# when I printed this shit, it gave me this ([25.72425621 38.59713548  0.45914788 61.67473243])   

cdf = pd.DataFrame (lm.coef_ , x.columns , columns = ['Coef'])

print (cdf) # I heard the higher the coeficeint the high the value 

# so basically NOW!! foe some weird reasons we are done with the modelm built and all that
#it seems too easy, but the children of god will survive.

#Unto the prediction model 

prediction = lm. predict(x_test) 

print (prediction) # This predicted a long list of numbers, like the anmount supposed to be spent becuase it's the same length as the dollars 


sns.scatterplot(x= prediction, y=y_test)
plt.xlabel("Predictions")
plt.title ("Damn it")
plt.show()


print ("Mean Absolute Error:", mean_absolute_error (y_test, prediction))
print ("Mean Sqaured Error:", mean_squared_error, (y_test, prediction) )
print ("RMSE:", math.sqrt (mean_squared_error (y_test, prediction))) # All this needs to be explained and how they be sued becuase girl!
# I don't know what this is but with time.

#Residuals 

residuals = y_test - prediction  # so getting it straight ressiduals are just the distance between the oredicted value to the actual value 

sns.displot (residuals, bins= 30 ,kde = True) 
plt.show ()



stats.probplot (residuals , dist= 'norm', plot = pylab) # Now I know this gives off the charp or something for confirmation
pylab.show()


    



