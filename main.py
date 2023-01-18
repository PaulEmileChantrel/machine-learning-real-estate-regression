import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
def load_data():
    df = pd.read_csv('../scraping/real_estate_data_v2p2.csv')
    return df

def main():
    
    #loading the data
    real_estate_df = load_data()

    # mini data cleaning
    real_estate_df = real_estate_df[real_estate_df['lat']!=0]
    real_estate_df = real_estate_df[real_estate_df['price']>=10000]

    # # first look at the data
    # print(real_estate_df.info())
    # print(real_estate_df.describe())
    # print(real_estate_df['bedrooms'].value_counts())
    # real_estate_df.hist(bins=50,figsize=(10,10))
    # plt.show()

    # spliting the data
    train_set,test_set = train_test_split(real_estate_df,test_size=0.2,random_state=42)
   
    # #ploting the real estate location
    # train_set['price_log'] = np.log(train_set['price'])
    # train_set.plot(kind='scatter',x='long',y='lat',alpha=0.3,c='price',cmap=plt.get_cmap("jet"),colorbar=True,)
    

    # from pandas.plotting import scatter_matrix

    # attributes = ['price','bedrooms','bathrooms','lat','long']
    # scatter_matrix(real_estate_df[attributes])

    #creation of new features 
    real_estate_df['total_bedrooms'] = real_estate_df['bedrooms']+real_estate_df['smaller_rooms']
    real_estate_df['total_rooms'] = real_estate_df['bathrooms']+real_estate_df['total_bedrooms']
    lat0,long0 = 43.651,-79.347 #DownTown Toronto 
    real_estate_df['r'] = np.sqrt((real_estate_df['lat']-lat0)**2+(real_estate_df['long']-long0)**2)



    plt.show()

if __name__ == '__main__':
    main()