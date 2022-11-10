##############################################IMPORT##################################################
#filepath = Path('zillow.csv')
#filepath.parent.mkdir(parents=True, exist_ok=True)
#df.to_csv(filepath, index=False)


# importing of all needed libraries and modules.  
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import warnings
warnings.filterwarnings("ignore")
from env import host, user, password
import wrangle
from scipy import stats
from scipy.stats import pearsonr, spearmanr

def get_db_url(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file 
    to create a connection url to access the codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
################################################### Acquire ###########################################
# use this function the 1st time to get initial dataset.
def new_zillow_data():
    '''This function reads in telco data from Codeup database.'''
    sql_query = """
                SELECT bedroomcnt, 
                bathroomcnt, 
                calculatedfinishedsquarefeet, 
                taxvaluedollarcnt, 
                yearbuilt, 
                taxamount, 
                fips
                FROM properties_2017 
                WHERE propertylandusetypeid = '261';
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    return df



def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv')
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df


##############################################PREP######################################################

def prep_zillow_data(df):

    #creat a column that is the age of the house 2017 minus the year it was built
    df['Age'] = 2017 - df['yearbuilt']

   # Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop all rows with any Null values, assign to df, and verify with df.info().
    df = df.dropna()
    
    #rename the columns so they are human readable
    df=df.rename(columns={"bedroomcnt":"Bedrooms","bathroomcnt":"Bathrooms",
                          "calculatedfinishedsquarefeet":"Square_Feet", 
                          "taxvaluedollarcnt": "Total_Home_Value","yearbuilt":"Year_Built",
                          "taxamount": "Taxes", "fips":"Region"})

    # Remove outliers
    df = (df[df.Bathrooms <= 5])
    df = df[df.Bedrooms <= 6] 
    df = df[df.Total_Home_Value < 2_000_000]
    df = df[df.Square_Feet < 10000]
    
    df.to_csv('clean_zillow.csv')
    
    return df

################################################### Split ##############################################

def split_zillow_data(df):
    '''
    This function performs split on zillow data
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test


###########################################wrangle####################################################
def wrangle_zillow():
    '''This function acquires, preps, and splits the zillow data'''
    
    df = get_zillow_data()
    
    df = prep_zillow_data(df)
    
    train, validate, test = split_zillow_data(df)
    
    return train, validate, test