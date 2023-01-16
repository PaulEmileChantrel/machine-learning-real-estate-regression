import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv('../scraping/real_estate_data_v2p2.csv')
    return df

def main():
    
    real_estate_df = load_data()

    # first look at the data
    print(real_estate_df.info())
    print(real_estate_df.describe())
    print(real_estate_df['bedrooms'].value_counts())
    real_estate_df.hist(bins=50,figsize=(10,10))
    plt.show()

    
if __name__ == '__main__':
    main()