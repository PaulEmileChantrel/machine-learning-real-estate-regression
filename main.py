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
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        bedrooms_idx,smaller_rooms_idx,bathrooms_idx = 3,4,5
     
        lat_idx,long_idx = 1,2
         
        lat0,long0 = 43.651,-79.347

        total_bedrooms = X[:,bedrooms_idx]+X[:,smaller_rooms_idx]
        total_rooms = total_bedrooms + X[:,bathrooms_idx]
        r = np.sqrt((X[:,lat_idx]-lat0)**2+(X[:,long_idx]-long0)**2)
        return np.c_[X,total_bedrooms,total_rooms,r]

def display_score(scores):
    print("Score:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())
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

    corr_matrix = real_estate_df.corr()
    print(corr_matrix["price"].sort_values(ascending=False))

    # spliting the data
    
    train_set,test_set = train_test_split(real_estate_df,test_size=0.2,random_state=42)
    real_estate_df = real_estate_df.reset_index(drop=True)
    labels = real_estate_df['price'].copy()
    real_estate_df = real_estate_df.drop(['price'],axis=1)
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
    
    #print(re_city_1hot.toarray())
    print(re_nums.info())
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num_pipeline = Pipeline([
        ('inputer',SimpleImputer(strategy="median")),
        ('attribd_adder',CombineAttributesAdder()),
        ('std_scaler',StandardScaler()),
    ])

    re_num_tr = num_pipeline.fit_transform(re_nums)
    #print(re_num_tr[:10])
    #plt.show()

    from sklearn.compose import ColumnTransformer

    num_attribs = list(re_nums)
    cat_attribs = ['city']

    full_pipeline = ColumnTransformer([
        ('num',num_pipeline,num_attribs),
        ('cat',OneHotEncoder(),cat_attribs),
    ])

    re_prepared = full_pipeline.fit_transform(real_estate_df)
    
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(re_prepared,labels)
    print(labels)
    some_data = real_estate_df.iloc[:10]
    some_labels = labels.iloc[:10]
    some_prepared_data = full_pipeline.transform(some_data)

    print("Predictions :", lin_reg.predict(some_prepared_data))
    print("Labels:",list(some_labels))
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(labels,lin_reg.predict(re_prepared))
    mse = np.sqrt(mse)
    print(mse)

    from sklearn.tree import DecisionTreeRegressor

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(re_prepared,labels)
    re_predictions = tree_reg.predict(re_prepared)
    tree_mse = mean_squared_error(labels,re_predictions)
    tree_mse = np.sqrt(tree_mse)
    print(tree_mse)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(tree_reg,re_prepared,labels,scoring='neg_mean_squared_error',cv=10)
    tree_rmse_score = np.sqrt(-scores)
    display_score(tree_rmse_score)

    lin_scores = cross_val_score(lin_reg,re_prepared,labels,scoring='neg_mean_squared_error',cv=10)
    lin_rmse_score = np.sqrt(-lin_scores)
    display_score(lin_rmse_score)

    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(re_prepared,labels)
    forest_prediction = forest_reg.predict(re_prepared)
    forest_rmse = mean_squared_error(labels,forest_prediction)
    forest_rmse = np.sqrt(forest_rmse)
    scores = cross_val_score(forest_reg,re_prepared,labels,scoring='neg_mean_squared_error',cv=10)
    forest_rmse_score = np.sqrt(-scores)
    display_score(forest_rmse_score)  

    #improving the model

    from sklearn.model_selection import GridSearchCV

    param_grid = [
        {'n_estimators': [3,10,30],'max_features': [2,4,6,8]},
        {'bootstrap': [False],'n_estimators': [3,10],'max_features': [2,3,4]},
    ]
    
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg,param_grid,cv=5,
                                scoring='neg_mean_squared_error',
                                return_train_score=True)

    grid_search.fit(re_prepared,labels)

    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    cvres = grid_search.cv_results_
    for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
        print(np.sqrt(-mean_score),params)

    features_importance = grid_search.best_estimator_.feature_importances_
    print(features_importance)

    extra_attribs = ["r","total_rooms","bathrooms"]
    cat_encoder = full_pipeline.named_transformers_['cat']
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + cat_one_hot_attribs
    print(sorted(zip(features_importance,attributes),reverse=True))

    final_model = grid_search.best_estimator_
    X_test = test_set.drop('price',axis=1)
    y_test = test_set['price'].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test,final_predictions)
    final_mse = np.sqrt(final_mse)

    print(final_mse)
      
    
if __name__ == '__main__':
    main()