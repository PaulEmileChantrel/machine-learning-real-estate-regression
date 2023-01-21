import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import traceback
def load_data():
    df = pd.read_csv('../scraping/real_estate_data_v2p2.csv')
    return df

def format_address(df):
    df = df[df['address']!='Address not available'].reset_index(drop=True)
    
    df[['street_address','unit_number','city','state']] = df['address'].str.split(',',3,expand=True)
   
    re_with_3row = df[df.state.isna()]
    re_with_4_row = df[df.state.notnull()]
    re_with_3row.rename(columns = {'unit_number':'state','state':'unit_number'},inplace=True)
    
    df = pd.concat([re_with_3row,re_with_4_row],ignore_index=True)


    return df

from sklearn.base import BaseEstimator,TransformerMixin

class CombineAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_berooms_per_rooms=True):
        self.add_berooms_per_rooms = add_berooms_per_rooms

    #todo
def main():
    
    #loading the data
    real_estate_df = load_data()

    # mini data cleaning
    real_estate_df = real_estate_df[real_estate_df['lat']!=0]
    real_estate_df = real_estate_df[real_estate_df['price']>=10000]

    # # first look at the data
    print(real_estate_df.info())
    # print(real_estate_df.describe())
    # print(real_estate_df['bedrooms'].value_counts())
    # real_estate_df.hist(bins=50,figsize=(10,10))
    # plt.show()

    
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

    #data pre-cleaning
    real_estate_df = real_estate_df[real_estate_df['bedrooms']!=0]
    real_estate_df = format_address(real_estate_df)

    # spliting the data
    
    train_set,test_set = train_test_split(real_estate_df,test_size=0.2,random_state=42)
   
    #print(real_estate_df[['street_address','unit_number','city','state','id']].head())
    
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy = "median")
    re_nums = real_estate_df.drop(['street_address','unit_number','city','state','link','image','address','posted','id'],axis=1)

    imputer.fit(re_nums)
    X = imputer.transform(re_nums)
    re_tr = pd.DataFrame(X,columns=re_nums.columns,index=re_nums.index)

    re_city = real_estate_df[['city']]
    # from sklearn.preprocessing import OrdinalEncoder
    # ordinal_encoder = OrdinalEncoder()
    # city_enc = ordinal_encoder.fit_transform(re_city)
    # print(city_enc[:10])
    from sklearn.preprocessing import OneHotEncoder
    city_encoder = OneHotEncoder()
    re_city_1hot = city_encoder.fit_transform(re_city)
    
    print(re_city_1hot.toarray())

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num_pipeline = Pipeline([
        ('inputer',SimpleImputer(strategy="median")),
        ('attribd_adder',CombineAttributesAdder()),
        ('std_scaler',StandardScaler()),
    ])

    #plt.show()

if __name__ == '__main__':
    main()