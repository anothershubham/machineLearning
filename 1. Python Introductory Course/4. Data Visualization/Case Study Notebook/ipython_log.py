# IPython log file

#import the libraries
import pandas as pd
import numpy as np
#read the dataset and check the first five rows
inp0 = pd.read_csv('googleplaystore_v2.csv')
inp0.head()
#Check the shape of the dataframe
inp0.shape
#Check the datatypes of all the columns of the dataframe
inp0.apply(lambda x: x.dtype)
#Check the number of null values in the columns
inp0.isnull().sum()
#Drop the rows having null values in the Rating field

inp0 = inp0[~inp0['Rating'].isnull()]
#Check the shape of the dataframe
inp0.shape
# Check the number of nulls in the Rating field again to cross-verify
inp0['Rating'].isnull().sum()
#Question
#Check the number of nulls in the dataframe again and find the total number of null values
inp0.isnull().sum()
#Inspect the nulls in the Android Version column
inp0[inp0['Android Ver'].isnull()]
#Drop the row having shifted values
# inp0.loc[10472]

inp0 = inp0[~((inp0['Android Ver'].isnull()) & (inp0['Category'] == '1.9'))]
inp0[inp0['Android Ver'].isnull()]
#Check the nulls againin Android version column to cross-verify
#Check the most common value in the Android version column
inp0['Android Ver'].describe()
inp0['Android Ver'].describe().top
#Fill up the nulls in the Android Version column with the above value
inp0['Android Ver'] = inp0['Android Ver'].fillna(inp0['Android Ver'].describe().top)
#Check the nulls in the Android version column again to cross-verify
inp0.isnull().sum()
#Check the nulls in the entire dataframe again
inp0[inp0['Current Ver'].isnull()]
#Check the most common value in the Current version column
inp0['Current Ver'].describe()
#Replace the nulls in the Current version column with the above value
inp0['Current Ver'] = inp0['Current Ver'].fillna(inp0['Current Ver'].describe().top)
inp0['Current Ver'].describe()
# Question : Check the most common value in the Current version column again
inp0['Current Ver'].describe()
inp1= inp0[inp0['Android Ver'] == '4.1 and up']
# pd.to_numeric(inp1['Price'])
inp1['Price'].value_counts()
#Check the datatypes of all the columns 
inp0.info()
#Question - Try calculating the average price of all apps having the Android version as "4.1 and up" 
#Analyse the Price column to check the issue
#Write the function to make the changes
inp0.Price = inp0.Price.apply(lambda x: 0 if x=="0" else float(x[1:]))
#Verify the dtype of Price once again
inp0.Price.value_counts()
#Analyse the Reviews column
inp0.isnull().sum()
inp0.Reviews.value_counts()
#Change the dtype of this column
inp0.Reviews = inp0.Reviews.apply(lambda x: int(x))
inp0.Reviews.describe()
#Check the quantitative spread of this dataframe
#Analyse the Installs Column
inp0.Installs.describe()
inp0.Installs.value_counts()
def desc(x):
    if(x[-1] == '+'):
        x = x[:-2]
    x = x.split(',')
    x = ''.join(x)
#     x = float(x)
    return x
inp0.Installs = inp0.Installs.apply(desc)
inp0.Installs.value_counts()
#Question Clean the Installs Column and find the approximate number of apps at the 50th percentile.
inp0.Installs.value_counts().describe()
#Perform the sanity checks on the Reviews column
#perform the sanity checks on prices of free apps 
#import the plotting libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#Create a box plot for the price column
# mtplt.boxplot(inp0.Price)
boxPriceCol = inp0.Price
boxPriceCol = boxPriceCol.sort_values()
boxPriceCol[boxPriceCol.shape[0]/2]
q1 = 0
q2 = 0
q3 = boxPriceCol[int(((boxPriceCol.shape[0]/2)+ boxPriceCol.shape[0])/2)]
q3
plt.boxplot(inp0.Price)
plt.show()
#Check the apps with price more than 200
inp0[inp0.Price > 200]
#Clean the Price column
inp0 = inp0[inp0.Price < 200]
inp0.describe()
#Create a box plot for paid apps
inp0[inp0.Price > 0].Price.plot.box()
#Check the apps with price more than 30
inp0[inp0.Price > 30]
#Clean the Price column again
inp0 = inp0[inp0.Price<=30]
inp0.Price.shape
# plt.boxplot(inp0.Price)
# plt.show()
# inp0.shape
inp0.Price.plot.box()
#Create a histogram of the Reviews
# ?plt.hist
plt.hist(inp0.Reviews, bins=5)
plt.show()
#Create a boxplot of the Reviews column

plt.boxplot(inp0.Reviews)
#Check records with 1 million reviews
inp0[inp0.Reviews >= 10000000]
#Drop the above records
inp0 = inp0[inp0.Reviews <= 1000000]
#Question - Create a histogram again and check the peaks

plt.hist(inp0.Reviews)
#Question - Create a box plot for the Installs column and report back the IQR
inp0.Installs = inp0.Installs.apply(lambda x: pd.to_numeric(x) if(type(x) != int) else x)
# inp0.Installs
# type(inp0.Installs[0])
# inp0.Installs.plot.box()
inp0.Installs.describe()
# inp0.Install
print(1.000000e+05 - 1.000000e+03)
#Question - CLean the Installs by removing all the apps having more than or equal to 100 million installs
inp0 = inp0[inp0.Installs <= 10000000]
inp0.shape
#Plot a histogram for Size as well.
inp0.Size.plot.hist()
#Question - Create a boxplot for the Size column and report back the median value
inp0.Size.plot.box()
#import the necessary libraries
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
#Create a distribution plot for rating
sns.boxplot(inp0.Rating)
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set_theme()
#Create a distribution plot for rating
sns.boxplot(inp0.Rating)
#Create a distribution plot for rating
sns.hist(inp0.Rating)
#Create a distribution plot for rating
sns.histplot(inp0.Rating)
#Change the number of bins
sns.histplot(inp0.Rating,bins=10)
#Change the number of bins
sns.histplot(inp0.Rating,bins=100)
#Change the number of bins
sns.histplot(inp0.Rating,bins=50)
#Change the number of bins
sns.histplot(inp0.Rating,bins=10)
#Create a distribution plot for rating
sns.displot(inp0.Rating)
#Change the number of bins
sns.displot(inp0.Rating,bins=10)
#Change the number of bins
sns.displot(inp0.Rating,rug=True)
#Change the number of bins
sns.displot(inp0.Rating,rug=False)
#Change the number of bins
sns.displot(inp0.Rating,rug=True, fit='norm')
#Change the number of bins
sns.distplot(inp0.Rating,rug=True, fit='norm')
#Change the number of bins
sns.distplot(inp0.Rating,rug=True, fit='norm')
#Change the number of bins
sns.distplot(inp0.Rating,rug=True, fit=norm)
#Change the number of bins
sns.distplot(inp0.Rating,kde=False)
#Change the number of bins
sns.distplot(inp0.Rating,bins=15)
#Change the number of bins
sns.distplot(inp0.Rating,bins=15,vertical=True)
#Change the colour of bins to green
sns.displot(inp0.Rating,bins=15,vertical=True)
#Change the colour of bins to green
sns.displot(inp0.Rating)
#Change the colour of bins to green
sns.distplot(inp0.Rating,bins=15,color='g')
#Change the colour of bins to green
sns.distplot(inp0.Rating,bins=15,color='g')
plt.show()
#Change the colour of bins to green
sns.distplot(inp0.Rating,bins=15,color='g')
#Analyse the Content Rating column
inp0.Rating
#Analyse the Content Rating column
inp0.Rating[inp0.Rating > 5]
#Analyse the Content Rating column
inp0.Rating[inp0.Rating > 5.0]
#Analyse the Content Rating column
inp0.Rating[inp0.Rating >= 5]
#Analyse the Content Rating column
inp0[inp0.Rating >= 5]
#Analyse the Content Rating column
inp0[inp0.Rating >= 5 && inp0.Installs > 1]
#Analyse the Content Rating column
inp0[inp0.Rating >= 5 & inp0.Installs > 1]
#Analyse the Content Rating column
inp0[inp0.Rating >= 5 and inp0.Installs > 1]
#Analyse the Content Rating column
inp0[inp0.Rating >= 5 & inp0.Installs > 1]
#Analyse the Content Rating column
inp0.Installs
#Analyse the Content Rating column
inp0[inp0.Rating >= 5 & inp0.Installs > 100]
#Analyse the Content Rating column
inp0[inp0.Installs > 100]
#Analyse the Content Rating column
inp0[inp0.Installs > 100 && inp0.Rating >= 5]
#Analyse the Content Rating column
inp0[inp0.Installs > 100 & inp0.Rating >= 5]
#Analyse the Content Rating column
inp0[inp0.Installs > 100 & inp0.Rating == 5]
#Analyse the Content Rating column
inp0[(inp0.Installs > 100) & (inp0.Rating == 5)]
#Analyse the Content Rating column
inp0[(inp0.Installs == 1) & (inp0.Rating == 5)]
#Analyse the Content Rating column
inp0[(inp0.Installs <= 10) & (inp0.Rating == 5)]
#Analyse the Content Rating column
inp0['Content Rating'].value_counts()
#Remove the rows with values which are less represented 
inp0 = inp0[~inp0['Content Rating'].isin(['Adults only 18+','Unrated'])]
#Reset the index
inp0['Content Rating'].value_counts()
#Reset the index
inp0['Content Rating'].value_counts()
inp0.reset_index(inplace=True,drop=True)
#Check the apps belonging to different categories of Content Rating 
inp0.info)
#Check the apps belonging to different categories of Content Rating 
inp0.info()
#Check the apps belonging to different categories of Content Rating 
inp0.info()
inp0['Content Rating'].value_counts()
#Check the apps belonging to different categories of Content Rating 
inp0.info()
inp0['Content Rating'].value_counts()
#Plot a pie chart
inp0['Content Rating'].plot.pie()
#Plot a pie chart
inp0['Content Rating'].value_counts().plot.pie()
#Plot a pie chart
inp0['Content Rating'].value_counts().plot.bar()
#Plot a pie chart
inp0['Content Rating'].value_counts().plot.pie()
#Plot a bar chart
inp0['Content Rating'].value_counts().plot.pie()
#Plot a bar chart
inp0['Content Rating'].value_counts().plot.bar()
#Plot a bar chart
inp0['Content Rating'].value_counts().plot.barh()
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
#Question - Plot a bar plot for checking the 4th highest Android version type
sns.barplot(inp0['Android Ver'])
#Question - Plot a bar plot for checking the 4th highest Android version type
sns.barplot(inp0['Android Ver'].value_counts())
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
console.log('Hello')
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
window.console.log('Hello')
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
get_ipython().run_line_magic('logon', '')
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
get_ipython().run_line_magic('logstart', '')
#import the libraries
import pandas as pd
import numpy as np
#import the libraries
import pandas as pd
import numpy as np
#import the libraries
import pandas as pd
import numpy as np
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
#import the libraries
import pandas as pd
import numpy as np
#read the dataset and check the first five rows
inp0 = pd.read_csv('googleplaystore_v2.csv')
inp0.head()
#Check the shape of the dataframe
inp0.shape
#Check the datatypes of all the columns of the dataframe
inp0.apply(lambda x: x.dtype)
#Check the number of null values in the columns
inp0.isnull().sum()
#Drop the rows having null values in the Rating field

inp0 = inp0[~inp0['Rating'].isnull()]
#Check the shape of the dataframe
inp0.shape
# Check the number of nulls in the Rating field again to cross-verify
inp0['Rating'].isnull().sum()
#Question
#Check the number of nulls in the dataframe again and find the total number of null values
inp0.isnull().sum()
#Inspect the nulls in the Android Version column
inp0[inp0['Android Ver'].isnull()]
#Drop the row having shifted values
# inp0.loc[10472]

inp0 = inp0[~((inp0['Android Ver'].isnull()) & (inp0['Category'] == '1.9'))]
inp0[inp0['Android Ver'].isnull()]
#Check the nulls againin Android version column to cross-verify
#Check the most common value in the Android version column
inp0['Android Ver'].describe()
inp0['Android Ver'].describe().top
#Fill up the nulls in the Android Version column with the above value
inp0['Android Ver'] = inp0['Android Ver'].fillna(inp0['Android Ver'].describe().top)
#Check the nulls in the Android version column again to cross-verify
inp0.isnull().sum()
#Check the nulls in the entire dataframe again
inp0[inp0['Current Ver'].isnull()]
#Check the most common value in the Current version column
inp0['Current Ver'].describe()
#Replace the nulls in the Current version column with the above value
inp0['Current Ver'] = inp0['Current Ver'].fillna(inp0['Current Ver'].describe().top)
inp0['Current Ver'].describe()
# Question : Check the most common value in the Current version column again
inp0['Current Ver'].describe()
inp1= inp0[inp0['Android Ver'] == '4.1 and up']
# pd.to_numeric(inp1['Price'])
inp1['Price'].value_counts()
#Check the datatypes of all the columns 
inp0.info()
#Question - Try calculating the average price of all apps having the Android version as "4.1 and up" 

#Analyse the Price column to check the issue

#Write the function to make the changes
inp0.Price = inp0.Price.apply(lambda x: 0 if x=="0" else float(x[1:]))
#Verify the dtype of Price once again
inp0.Price.value_counts()
#Analyse the Reviews column
inp0.isnull().sum()
inp0.Reviews.value_counts()
#Change the dtype of this column
inp0.Reviews = inp0.Reviews.apply(lambda x: int(x))
inp0.Reviews.describe()
#Check the quantitative spread of this dataframe

#Analyse the Installs Column
inp0.Installs.describe()
inp0.Installs.value_counts()
def desc(x):
    if(x[-1] == '+'):
        x = x[:-2]
    x = x.split(',')
    x = ''.join(x)
#     x = float(x)
    return x
inp0.Installs = inp0.Installs.apply(desc)
inp0.Installs.value_counts()
#Question Clean the Installs Column and find the approximate number of apps at the 50th percentile.
inp0.Installs.value_counts().describe()
#Perform the sanity checks on the Reviews column
#perform the sanity checks on prices of free apps 
#import the plotting libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#Create a box plot for the price column
# mtplt.boxplot(inp0.Price)
boxPriceCol = inp0.Price
boxPriceCol = boxPriceCol.sort_values()
boxPriceCol[boxPriceCol.shape[0]/2]
q1 = 0
q2 = 0
q3 = boxPriceCol[int(((boxPriceCol.shape[0]/2)+ boxPriceCol.shape[0])/2)]
q3
plt.boxplot(inp0.Price)
plt.show()
#Check the apps with price more than 200
inp0[inp0.Price > 200]
#Clean the Price column
inp0 = inp0[inp0.Price < 200]
inp0.describe()
#Create a box plot for paid apps
inp0[inp0.Price > 0].Price.plot.box()
#Check the apps with price more than 30
inp0[inp0.Price > 30]
#Clean the Price column again
inp0 = inp0[inp0.Price<=30]
inp0.Price.shape
# plt.boxplot(inp0.Price)
# plt.show()
# inp0.shape
inp0.Price.plot.box()
#Create a histogram of the Reviews
# ?plt.hist
plt.hist(inp0.Reviews, bins=5)
plt.show()
#Create a boxplot of the Reviews column

plt.boxplot(inp0.Reviews)
#Check records with 1 million reviews
inp0[inp0.Reviews >= 10000000]
#Drop the above records
inp0 = inp0[inp0.Reviews <= 1000000]
#Question - Create a histogram again and check the peaks

plt.hist(inp0.Reviews)
#Question - Create a box plot for the Installs column and report back the IQR
inp0.Installs = inp0.Installs.apply(lambda x: pd.to_numeric(x) if(type(x) != int) else x)
# inp0.Installs
# type(inp0.Installs[0])
# inp0.Installs.plot.box()
inp0.Installs.describe()
# inp0.Install
print(1.000000e+05 - 1.000000e+03)
#Question - CLean the Installs by removing all the apps having more than or equal to 100 million installs
inp0 = inp0[inp0.Installs <= 10000000]
inp0.shape
#Plot a histogram for Size as well.
inp0.Size.plot.hist()
#Question - Create a boxplot for the Size column and report back the median value
inp0.Size.plot.box()
#import the necessary libraries
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set_theme()
#Create a distribution plot for rating
sns.displot(inp0.Rating)
#Change the number of bins
sns.distplot(inp0.Rating,bins=15,vertical=True)
#Change the colour of bins to green
sns.distplot(inp0.Rating,bins=15,color='g')
#Apply matplotlib functionalities

#Check all the styling options
#Change the number of bins to 20
#Analyse the Content Rating column
inp0['Content Rating'].value_counts()
#Remove the rows with values which are less represented 
inp0 = inp0[~inp0['Content Rating'].isin(['Adults only 18+','Unrated'])]
#Reset the index
inp0.reset_index(inplace=True,drop=True)
#Check the apps belonging to different categories of Content Rating 
inp0.info()
inp0['Content Rating'].value_counts()
#Plot a pie chart
inp0['Content Rating'].value_counts().plot.pie()
#Plot a bar chart
inp0['Content Rating'].value_counts().plot.barh()
#Question - Plot a bar plot for checking the 4th highest Android version type
inp0['Android Ver'].value_counts().plot.barh()
###Size vs Rating
sns.scatterplot(inp0.Rating)
##Plot a scatter-plot in the matplotlib way between Size and Rating
###Size vs Rating
sns.scatterplot(inp0.Rating.values)
##Plot a scatter-plot in the matplotlib way between Size and Rating
###Size vs Rating
sns.scatterplot(inp0)
##Plot a scatter-plot in the matplotlib way between Size and Rating
###Size vs Rating
sns.scatterplot(inp0.Rating,inp0.Installs)
##Plot a scatter-plot in the matplotlib way between Size and Rating
###Size vs Rating
sns.scatterplot(inp0.Size,inp0.Rating)
##Plot a scatter-plot in the matplotlib way between Size and Rating
### Plot the same thing now using a jointplot
sns.jointplot(inp0.Size,inp0.Rating)
### Plot the same thing now using a jointplot
sns.set_theme('default')
sns.jointplot(inp0.Size,inp0.Rating)
### Plot the same thing now using a jointplot
sns.set_theme(style='default')
sns.jointplot(inp0.Size,inp0.Rating)
### Plot the same thing now using a jointplot
sns.set_theme(style='white')
sns.jointplot(inp0.Size,inp0.Rating)
### Plot the same thing now using a jointplot
import scipy.stats as stat
sns.set_theme(style='white')
sns.jointplot(inp0.Size,inp0.Rating, stat_func=stat.pearsonr)
### Plot the same thing now using a jointplot
import scipy.stats as stat
sns.set_theme(style='white')
sns.jointplot('Size','Rating', data=inp0, stat_func=stat.pearsonr)
### Plot the same thing now using a jointplot
import scipy.stats as stat
sns.set_theme(style='white')
sns.jointplot(inp0.Size,inp0.Rating,stat_func=stat.pearsonr)
### Plot the same thing now using a jointplot
import scipy.stats as stat
sns.set_theme(style='white')
sns.jointplot(inp0.Size,inp0.Rating).annotate(stats.pearsonr)
### Plot the same thing now using a jointplot
import scipy.stats as stat
sns.set_theme(style='white')
sns.set(style="darkgrid", color_codes=True)
sns.jointplot(inp0.Size,inp0.Rating).annotate(stats.pearsonr)
### Plot the same thing now using a jointplot
import scipy.stats as stat
sns.set_theme(style='white')
sns.set(style="darkgrid", color_codes=True)
sns.jointplot(inp0.Size,inp0.Rating).annotate(stats.pearsonr)
plt.show()
### Plot the same thing now using a jointplot
sns.jointplot(inp0.Size,inp0.Rating)
plt.show()
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,height=200)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,height=100)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,height=50)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,height=50, ratio=10)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,height=20, ratio=10)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,height=10, ratio=10)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,height=10)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,dist=False)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,hist=False)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,kde=False)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating)
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Price,inp0.Rating,kind='kde')
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Size,inp0.Rating,kind='kde',color='g')
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Size,inp0.Rating,kind='kde',color='G')
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Size,inp0.Rating,kind='kde',color='G')
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Size,inp0.Rating,kind='kde',color='g')
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Size,inp0.Rating,kind='kde',color='G')
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Size,inp0.Rating,kind='kde',color='g')
## Plot a jointplot for Price and Rating
sns.set_style('white')
sns.jointplot(inp0.Size,inp0.Rating,kind='kde',color='g')
## Plot a jointplot for Price and Rating
sns.jointplot(inp0.Size,inp0.Rating,kind='kde',color='g')
##Plot a reg plot for Price and Rating and observe the trend
sns.jointplot(inp0.Price,inp0.Rating, kind='kde')
##Plot a reg plot for Price and Rating and observe the trend
sns.jointplot(inp0.Price,inp0.Rating, kind='reg')
## Question - Plot a reg plot for Price and Rating again for only the paid apps.
# inp1 = inp0[in]
# sns.jointplot(inp0.Price,inp0.Rating, kind='reg')
inp0
## Question - Plot a reg plot for Price and Rating again for only the paid apps.
inp1 = inp0[inp0.Type != 'Free']
sns.jointplot(inp0.Price,inp0.Rating, kind='reg')
## Question - Plot a reg plot for Price and Rating again for only the paid apps.
inp1 = inp0[inp0.Type != 'Free']
sns.jointplot(inp1.Price,inp1.Rating, kind='reg')
## Question - Plot a reg plot for Price and Rating again for only the paid apps.
inp1 = inp0[inp0.Type != 'Free']
inp1
sns.jointplot(inp1.Price,inp1.Rating, kind='reg')
## Question - Plot a reg plot for Price and Rating again for only the paid apps.
inp1 = inp0[inp0.Type != 'Free']
print(inp1)
sns.jointplot(inp1.Price,inp1.Rating, kind='reg')
## Question - Plot a reg plot for Price and Rating again for only the paid apps.
inp1 = inp0[inp0.Type != 'Free']
sns.jointplot(inp1.Price,inp1.Rating, kind='reg')
## Create a pair plot for Reviews, Size, Price and Rating

sns.pairplot(inp0[['Reviews','Price','Size','Rating']])
##Plot a bar plot of Content Rating vs Average Rating 
sns.barplot(inp0,x='Content Rating', y='Rating')
##Plot a bar plot of Content Rating vs Average Rating 
sns.barplot(inp0,x='Content Ratings', y='Rating')
## Create a pair plot for Reviews, Size, Price and Rating

sns.pairplot(inp0[['Reviews','Price','Size','Rating']])
inp0
## Create a pair plot for Reviews, Size, Price and Rating

sns.pairplot(inp0[['Reviews','Price','Size','Rating']])
##Plot a bar plot of Content Rating vs Average Rating 
sns.barplot(inp0,x='Content Rating', y='Rating')
##Plot a bar plot of Content Rating vs Average Rating 
sns.barplot(inp0,x='Content Rating', y="Rating")
##Plot a bar plot of Content Rating vs Average Rating 
sns.barplot(inp0,x='Content Rating', y"Rating")
##Plot a bar plot of Content Rating vs Average Rating 
sns.barplot(inp0,x='Content Rating', y="Rating")
##Plot a bar plot of Content Rating vs Average Rating 
sns.barplot(data=inp0,x='Content Rating', y="Rating")
##Plot the bar plot again with Median Rating
sns.barplot(data=inp0,x='Content Rating',y='Rating', estimator='median')
##Plot the bar plot again with Median Rating
sns.barplot(data=inp0,x='Content Rating',y='Rating', estimator="meadin")
##Plot the bar plot again with Median Rating
sns.barplot(data=inp0,x='Content Rating',y='Rating', estimator="median")
##Plot the bar plot again with Median Rating
sns.barplot(data=inp0,x='Content Rating',y='Rating', estimator=np.quantile(0.05))
##Plot the bar plot again with Median Rating
sns.barplot(data=inp0,x='Content Rating',y='Rating', estimator=np.median)
##Plot the above bar plot using the estimator parameter
sns.barplot(data=inp0, x='Content Rating', y='Rating', estimator=lambda x: np.quantile(x,0.05))
##Question - Plot the bar plot with the minimum Rating
sns.barplot(data=inp0, x='Content Rating', y='Rating', estimator=lambda x: np.minimum)
##Question - Plot the bar plot with the minimum Rating
sns.barplot(data=inp0, x='Content Rating', y='Rating', estimator=lambda x: np.min)
##Question - Plot the bar plot with the minimum Rating
sns.barplot(data=inp0, x='Content Rating', y='Rating', estimator=lambda x: np.min_scalar_type)
##Question - Plot the bar plot with the minimum Rating
sns.barplot(data=inp0, x='Content Rating', y='Rating', estimator=lambda x: np.minimum())
##Question - Plot the bar plot with the minimum Rating
sns.barplot(data=inp0, x='Content Rating', y='Rating', estimator=np.min)
##Plot a box plot of Rating vs Content Rating
sns.boxplot(data=inp0, x='Content Rating', y='Rating')
##Plot a box plot of Rating vs Content Rating
sns.figsize([7.8])
sns.boxplot(data=inp0, x='Content Rating', y='Rating')
##Plot a box plot of Rating vs Content Rating
sns.figure(figsize=[7.8])
sns.boxplot(data=inp0, x='Content Rating', y='Rating')
##Plot a box plot of Rating vs Content Rating
plt.figure(figsize=[7.8])
sns.boxplot(data=inp0, x='Content Rating', y='Rating')
##Plot a box plot of Rating vs Content Rating
plt.figure(figsize=[7,8])
sns.boxplot(data=inp0, x='Content Rating', y='Rating')
##Plot a box plot of Rating vs Content Rating
plt.figure(figsize=[9,10])
sns.boxplot(data=inp0, x='Content Rating', y='Rating')
##Question - Plot a box plot for the Rating column only
sns.boxplot(data=inp0, x='Content Rating')
##Question - Plot a box plot for the Rating column only
sns.boxplot(data=inp0, 'Content Rating')
##Question - Plot a box plot for the Rating column only
sns.boxplot(data=inp0, ax='Content Rating')
##Question - Plot a box plot for the Rating column only
sns.boxplot(data=inp0['Rating'])
##Question - Plot a box plot of Ratings across the 4 most popular Genres
sns.boxplot(data=inp0, x='Genres', y= 'Rating')
##Question - Plot a box plot of Ratings across the 4 most popular Genres
sns.boxplot(data=inp0, x='Genres', y= 'Rating')
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
sns.boxplot(data=inp0, x='Genres', y= 'Rating')
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].describe()
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].describe().top()
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].describe()
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
sns.boxplot(data=inp0, x='Genres', y= 'Rating')
# inp0['Genres']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[0]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[0].key
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()['Tools']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[:4]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[:4].index
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[:4].index.toList()
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[:4].index.tolist()
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0[inp0.Genres=inp0['Genres'].value_counts()[:4].index.tolist()]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0[inp0.Genres==inp0['Genres'].value_counts()[:4].index.tolist()]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0[inp0.Genres===inp0['Genres'].value_counts()[:4].index.tolist()]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0[inp0.Genres==inp0['Genres'].value_counts()[:4].index.tolist()]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[:4].index.tolist()
inp0[inp0.Genres='']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[:4].index.tolist()
inp0[inp0.Genres='']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[:4].index.tolist()
# inp0[inp0.Genres='']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
# inp0['Genres'].value_counts()[:4].index.tolist()
inp0[inp0.Genres='Tools']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
# inp0['Genres'].value_counts()[:4].index.tolist()
inp0[inp0.Genres=='Tools']
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
# inp0['Genres'].value_counts()[:4].index.tolist()
inp0[inp0.Genres==['Tools']]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
# inp0['Genres'].value_counts()[:4].index.tolist()
inp0[inp0.Genres.isin(inp0['Genres'].value_counts()[:4].index.tolist())]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
inp0['Genres'].value_counts()[:4].index.tolist()
# inp0[inp0.Genres.isin(inp0['Genres'].value_counts()[:4].index.tolist())]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
# inp0['Genres'].value_counts()[:4].index.tolist()
inp0[inp0.Genres.isin(inp0['Genres'].value_counts()[:4].index.tolist())]
##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,10])
# sns.boxplot(data=inp0, x='Genres', y= 'Rating')
# inp0['Genres'].value_counts()[:4].index.tolist()
inp1 = inp0[inp0.Genres.isin(inp0['Genres'].value_counts()[:4].index.tolist())]
sns.boxplot(data=inp1, x='Genres',y='Rating')
##Ratings vs Size vs Content Rating
pd.qcut(inp0,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

##Ratings vs Size vs Content Rating
pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

##Ratings vs Size vs Content Rating
inp0.Size_Bucket = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

##Ratings vs Size vs Content Rating
inp0.Size_Bucket = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

inp0
##Ratings vs Size vs Content Rating
inp0.Size_Bucket = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

inp0
##Ratings vs Size vs Content Rating
inp0.Size_Bucket = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

inp0
##Ratings vs Size vs Content Rating
inp0.Size_Bucket = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

inp0
##Ratings vs Size vs Content Rating
inp0.Size_Bucket = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

inp0
##Ratings vs Size vs Content Rating
inp0['Size_Bucket'] = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

inp0
##Ratings vs Size vs Content Rating
inp0['Size_Bucket'] = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

inp0.Size_Bucket.value_counts()
##Ratings vs Size vs Content Rating
inp0['Size_Bucket'] = pd.qcut(inp0.Size,5,['VL','L','M','H','VH'])
##Prepare buckets for the Size column using pd.qcut

inp0
##Create a pivot table for Size_buckets and Content Rating with values set to Rating
pd.pivot_table(data=inp0, values=['Rating'], index='Content Rating', columns='Size_Buckets')
##Create a pivot table for Size_buckets and Content Rating with values set to Rating
pd.pivot_table(data=inp0, values=['Rating'], index='Content Rating', columns='Size_Bucket')
##Create a pivot table for Size_buckets and Content Rating with values set to Rating
pd.pivot_table(data=inp0, values='Rating', index='Content Rating', columns='Size_Bucket')
##Change the aggregation to median
pd.pivot_table(data=inp0, values='Rating', index='Content Rating', columns='Size_Bucket',aggfunc='median')
##Change the aggregation to 20th percentile
pd.pivot_table(data=inp0, values='Rating', index='Content Rating', columns='Size_Bucket',aggfunc=lambda x: np.quantile(x,0.2))
##Store the pivot table in a separate variable
pd.pivot_table(data=inp0, values='Rating', index='Content Rating', columns='Size_Bucket',aggfunc=lambda x: np.quantile(x,0.2))tmp = 
##Store the pivot table in a separate variable
tmp = pd.pivot_table(data=inp0, values='Rating', index='Content Rating', columns='Size_Bucket',aggfunc=lambda x: np.quantile(x,0.2))
##Plot a heat map
sns.heatmap(tmp)
##Apply customisations
sns.heatmap(tmp,cmap='Greens',annot=True)
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
inp0
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
pd.qcut(inp0,5)
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
pd.qcut(inp0.Review,5)
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
pd.qcut(inp0.Reviews,5)
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
pd.qcut(inp0.Reviews,5,labels=['VL','L','M','H','VH'])
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
inp0['Review Bucket'] = pd.qcut(inp0.Reviews,5,labels=['VL','L','M','H','VH'])
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
inp0['Review Bucket'] = pd.qcut(inp0.Reviews,5,labels=['VL','L','M','H','VH'])
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
inp0['Review Bucket'] = pd.qcut(inp0.Reviews,5,labels=['VL','L','M','H','VH'])
inp0
pd.pivot_table(inp0,values='Rating',index='Content Rating', columns='Review Bucket')
##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating
inp0['Review Bucket'] = pd.qcut(inp0.Reviews,5,labels=['VL','L','M','H','VH'])
pd.pivot_table(inp0,values='Rating',index='Content Rating', columns='Review Bucket',aggfunc=np.median)
tmp1 = pd.pivot_table(inp0,values='Rating',index='Content Rating', columns='Review Bucket',aggfunc=np.median)
sns.heatmap(tmp1)
## Extract the month from the Last Updated Date
pd.to_datetime(inp0['Last Updated'])
## Extract the month from the Last Updated Date
pd.to_datetime(inp0['Last Updated']).dt.month
## Extract the month from the Last Updated Date
pd.to_datetime(inp0['Last Updated'])
## Extract the month from the Last Updated Date
pd.to_datetime(inp0['Last Updated'])
inp0['Last Updated']
## Extract the month from the Last Updated Date
pd.to_datetime(inp0['Last Updated']).dt.month
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0['Updated Month'].value_counts()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0['Updated Month'].value_counts().sort()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
sorted(inp0['Updated Month'].value_counts())
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
sorted(inp0['Updated Month'].value_counts(),key=inp0['Updated Month'].value_counts().index)
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
sorted(inp0['Updated Month'].value_counts(),key=inp0['Updated Month'].value_counts().index)
inp0['Updated Month'].value_counts().index
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
# sorted(inp0['Updated Month'].value_counts(),key=inp0['Updated Month'].value_counts().index)
inp0['Updated Month'].value_counts().index
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
# sorted(inp0['Updated Month'].value_counts(),key=inp0['Updated Month'].value_counts().index)
inp0['Updated Month'].value_counts().index.tolist()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
sorted(inp0['Updated Month'].value_counts(),key=inp0['Updated Month'].value_counts().index.tolist())
inp0['Updated Month'].value_counts().index.tolist()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
sorted(inp0['Updated Month'].value_counts(),key=inp0['Updated Month'].value_counts().index.tolist())
# inp0['Updated Month'].value_counts().index.tolist()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0['Updated Month'].value_counts().index.tolist()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0['Updated Month'].plot()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0.groupby(['Updated Month'])['Rating']
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0.groupby(['Updated Month'])['Rating'].plot()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0.groupby(['Updated Month'])['Rating'].mean.plot()
## Extract the month from the Last Updated Date
inp0['Updated Month'] = pd.to_datetime(inp0['Last Updated']).dt.month
inp0.groupby(['Updated Month'])['Rating'].mean().plot()
## Find the average Rating across all the months
inp0.groupby(['Updated Month'])['Rating'].mean().plot()
## Find the average Rating across all the months
inp0.groupby(['Updated Month'])['Rating'].avg().plot()
## Find the average Rating across all the months
inp0.groupby(['Updated Month'])['Rating'].mean().plot()
## Create a pivot table for Content Rating and updated Month with the values set to Installs
pd.pivot_table(data=inp0,values='Installs',columns='Content Rating',index='Updated Month')
## Create a pivot table for Content Rating and updated Month with the values set to Installs
pd.pivot_table(data=inp0,values='Installs',columns='Content Rating',index='Updated Month',aggfunc=sum)
## Create a pivot table for Content Rating and updated Month with the values set to Installs
pd.pivot_table(data=inp0,values='Installs',columns='Content Rating',index='Updated Month')
## Create a pivot table for Content Rating and updated Month with the values set to Installs
pd.pivot_table(data=inp0,values='Installs',columns='Content Rating',index='Updated Month',aggfunc=sum)
##Store the table in a separate variable
monthly = pd.pivot_table(data=inp0,values='Installs',columns='Content Rating',index='Updated Month',aggfunc=sum)
##Plot the stacked bar chart.
monthly.plot(kind='bar',stacked=True)
##Plot the stacked bar chart.
monthly.plot(kind='bar',stacked=True,figsize=[10,8])
##Plot the stacked bar chart again wrt to the proportions.
monthly(['Everyone','Everyone 10+','Mature 17+','Teen']).apply(lambda x: x/x.sum())
##Plot the stacked bar chart again wrt to the proportions.
monthly[['Everyone','Everyone 10+','Mature 17+','Teen']].apply(lambda x: x/x.sum())
##Plot the stacked bar chart again wrt to the proportions.
monthly_perc = monthly[['Everyone','Everyone 10+','Mature 17+','Teen']].apply(lambda x: x/x.sum())
monthly_perc.plot(kind='bar',stacked=True,figsize=[10,8])
##Plot the stacked bar chart again wrt to the proportions.
monthly_perc = monthly[['Everyone','Everyone 10+','Mature 17+','Teen']].apply(lambda x: x/x.sum(),axis=1)
##Plot the stacked bar chart again wrt to the proportions.
monthly_perc = monthly[['Everyone','Everyone 10+','Mature 17+','Teen']].apply(lambda x: x/x.sum(),axis=1)
monthly_perc.plot(kind='bar',stacked=True,figsize=[10,8])
#Take the table you want to plot in a separate variable
res = inp0.groupby(['Updated Month'])['Rating'].mean()
#Take the table you want to plot in a separate variable
inp0.groupby(['Updated Month'])['Rating'].mean()
#Import the plotly libraries
res = res.reset_index()
#Import the plotly libraries
res = res.reset_index()
import plotly.express as ps
#Prepare the plot
ps.line(res,x='Updated Month', y='Rating',title='Monthly Average Rating')
#Prepare the plot
fig = ps.line(res,x='Updated Month', y='Rating',title='Monthly Average Rating')
#Prepare the plot
fig = ps.line(res,x='Updated Month', y='Rating',title='Monthly Average Rating')
fig.show()
