from typing import Pattern
import findspark
findspark.init()
findspark.find()

import pyspark

from pyspark.sql import *
from pyspark.sql.functions import *
import datetime
import pyspark.sql.functions as F 

spark = SparkSession.builder \
            .master("local") \
            .appName("Book Recommendation System") \
            .getOrCreate()

df_taxi = spark.read.csv('Contacts.csv', inferSchema=True, header=True)

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import TimestampType
from spellchecker import SpellChecker

  #******** great expectation***********  
      
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import great_expectations as ge
from great_expectations.dataset.sparkdf_dataset import SparkDFDataset

 #*****************************************

def preprocessing(data_all_df):
    data_all_df = data_all_df.withColumn("createdon",data_all_df['createdon'].cast(TimestampType()))
    data_all_df = data_all_df.withColumn("modifiedon",data_all_df['modifiedon'].cast(TimestampType()))

    return data_all_df



def dataprofile(data_all_df,data_cols):
    native_data = data_all_df
    data_all_df = preprocessing(data_all_df)
    data_df = data_all_df.select(data_cols)
    columns2Bprofiled = data_df.columns
    global schema_name, table_name
    if not 'schema_name' in globals():
        schema_name = 'schema_name'
    if not 'table_name' in globals():
        table_name = 'table_name' 
    #to_date(col("createdon"), "yyy-MM-dd")

    dprof_df = pd.DataFrame({'schema_name':[schema_name] * len(data_df.columns),\
                             'table_name':[table_name] * len(data_df.columns),\
                             'column_names':data_df.columns,\
                             'data_types':[x[1] for x in data_df.dtypes]}) 
    dprof_df = dprof_df[['schema_name','table_name','column_names', 'data_types']]
    #dprof_df.set_index('column_names', inplace=True, drop=False)
    # ======================
    num_rows = data_df.count()
    dprof_df['num_rows'] = num_rows

    # ======================    
    # number of rows with nulls and nans   
    df_nacounts = data_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data_df.columns \
                                  if data_df.select(c).dtypes[0][1]!='timestamp']).toPandas().transpose()
    
    df_nacounts = df_nacounts.reset_index()  
    df_nacounts.columns = ['column_names','num_null']
    dprof_df = pd.merge(dprof_df, df_nacounts, on = ['column_names'], how = 'left')
      
    # ========================
    #Completeness

    dprof_df['Completeness'] = ((dprof_df['num_null'] / num_rows)*100)

      
    # ========================
    #Timeliness

    datatemp=data_df.withColumn("Timeliness", ((current_timestamp().cast("long")-data_df['modifiedon'].cast("long"))/(current_timestamp().cast("long")-data_df['createdon'].cast("long")))*100)    
    Timeliness = datatemp.agg(F.sum("Timeliness")).collect()[0][0]/num_rows
    dprof_df['Timeliness'] = Timeliness

    # ========================
    #Volatility

    datatemp=data_df.withColumn("Volatility", ((data_df['modifiedon'].cast("long")-data_df['createdon'].cast("long"))/(current_timestamp().cast("long")-data_df['createdon'].cast("long")))*100)    
    Volatility = datatemp.agg(F.sum("Volatility")).collect()[0][0]/num_rows
    dprof_df['Volatility'] = Volatility

    # ========================
    #Uniqueness

    datatemp= data_df[data_df.columns[1:]].distinct()
    Uniqueness = (datatemp.count()/num_rows)*100
    dprof_df['Uniqueness'] = Uniqueness

    


     # ========================
    #Ease of Manipulation


    Types = ['id', 'nom', 'prenom', 'age', 'createdon', 'modifiedon']
    df = pd.read_csv('Contacts.csv')
    pre_df = pd.read_csv('pre.csv')
    missing_columns = (len(df.columns)-len(pre_df.columns))/len(df.columns)
    df[Types].compare(pre_df[Types])
    changed_columns = (len( (df[Types].compare(pre_df[Types])).columns)/2)/len(df.columns)

    Ease = ((missing_columns + changed_columns)/2)*100
    dprof_df['Ease'] = Ease
    

    


    # ========================
    #Readability

    spell  = SpellChecker()
    df = pd.read_csv('Contacts.csv')
    unreadable_data = [100] * (len(data_cols))


    def spell_check(x):
        correct_word = []
        mispelled_word = x.split()
        for word in mispelled_word:
            correct_word.append(spell.correction(word))
        return ' '.join(correct_word)
    
    #df['spell_corrected_sentence'] = df['prenom'].apply(lambda x: spell_check(x))

    
    my_df_pd = ge.dataset.PandasDataset(df)
    for i in range(0,len(data_cols)):
        if data_df.select(data_cols[i]).dtypes[0][1] =='string':
            df[data_cols[i]] = df[data_cols[i]].astype(str)
            df['spell_corrected_sentence'] = df[data_cols[i]].apply(lambda x: spell_check(x))
            result = my_df_pd.expect_column_pair_values_to_be_equal(data_cols[i], 'spell_corrected_sentence')
            unreadable_data[i] = 100-result['result']['unexpected_percent']

    dprof_df['Readability'] = unreadable_data




    # ========================
    #Accessibility


     # ========================
    #Relevancy


    
    import numpy as np
    access = np.array([2,5,7,8,9,8,9])
    total = np.sum(access)
    relevant_data = [0] * (len(data_cols))
    for i in range(len(data_cols)):
        relevant_data[i]= (access[i]/total)*100

    dprof_df['Relevancy'] = relevant_data

    # ========================
    #Consistency  
    # 
    # seefavoris corrupted records  

    Types = ['int', 'S', 'S', 'int', 'M', 'M', 'int']
    inconsistent_data = [0] * (len(data_cols))
    df_taxi_pd = pd.read_csv('Contacts.csv')
    my_df_pd = ge.dataset.PandasDataset(df_taxi_pd)
    for i in range(1,len(data_cols)):
        result = my_df_pd.expect_column_values_to_be_of_type(data_cols[i], Types[i])
        inconsistent_data[i] = 100-result['result']['missing_percent']

    dprof_df['Consistency'] = inconsistent_data
   


    # ========================
    #Conformity

    Patterns = ['^[0-9]*$','^[0-9]*$','^[0-9]*$','^[0-9]*$','^[0-9]*$','^[0-9]*$', '^[0-9]*$']
    Conform_data = [0] * (len(data_cols))
    for i in range(len(data_cols)-1):
        Conform_data[i] = ((data_df.filter(col(data_cols[i]).rlike(Patterns[i])).count())/ num_rows)*100

    dprof_df['Conformity'] = Conform_data


    # ========================
    #Intergrety

    df = pd.read_csv('Contacts.csv')
    pre_df = pd.read_csv('pre.csv')

    df.eq(pre_df)
    Integrety = df.eq(pre_df).sum().sum()/df.eq(pre_df).size
    dprof_df['Intergrety'] = (df.eq(pre_df).sum().sum()/df.eq(pre_df).size)*100


     # ========================
    #Measure Mean Value for dimensions

    Completeness_mtr = dprof_df['Completeness'].mean() 
    Timeliness_mtr = dprof_df['Timeliness'].mean() 
    Volatility_mtr = dprof_df['Volatility'].mean() 
    Uniqueness_mtr = dprof_df['Uniqueness'].mean() 
    Ease_mtr = dprof_df['Ease'].mean() 
    Readability_mtr = dprof_df['Readability'].mean() 
    Relevancy_mtr = dprof_df['Relevancy'].mean() 
    Consistency_mtr = dprof_df['Consistency'].mean() 
    Conformity_mtr = dprof_df['Conformity'].mean() 
    Integrity_mtr= dprof_df['Intergrety'].mean()
    Security_mtr=40
    Accessibility_mtr=100
 

    # ========================
    #Quality Aspects

    Reliability= 0.7*Integrity_mtr+0.3*Volatility_mtr
    Availability= 0.8*Security_mtr+0.2*Accessibility_mtr
    Usability= 0.5*Completeness_mtr+0.3*Relevancy_mtr+0.2*Ease_mtr
    Pertinence= 0.7*Timeliness_mtr+0.3*Uniqueness_mtr
    Validity=0.4*Consistency_mtr+0.2*Readability_mtr+0.4*Conformity_mtr


    # ========================
    #Quality Score

    Quality=0.3*Reliability+0.1*Availability+0.2*Usability+0.1*Pertinence+0.3*Validity
    print(Quality)
    

     # ========================
    #Graph

    import numpy as np
    import matplotlib.pyplot as plt
    x_1=[1,2,3,4,5,6]
    diemnsions_1=[Volatility_mtr,Consistency_mtr,Timeliness_mtr,Conformity_mtr, Security_mtr, Completeness_mtr]
    labels_dim_1=['Volatility', 'Consistency', 'Timeliness', 'Conformity', 'Security', 'Completeness']
    plt.bar(x_1,diemnsions_1, tick_label=labels_dim_1, width=0.5)
    plt.xlabel("Quality Metrics")
    plt.ylabel("Score")
    plt.show()

    x_2=[1,2,3,4,5,6]
    diemnsions_2=[Uniqueness_mtr, Ease_mtr, Readability_mtr, Accessibility_mtr, Integrity_mtr, Relevancy_mtr]
    labels_dim_2=['Uniqueness', 'Ease', 'Readability', 'Accessibility', 'Integrity','Relevancy']
    plt.bar(x_2,diemnsions_2, tick_label=labels_dim_2, width=0.5)
    plt.xlabel("Quality Metrics")
    plt.ylabel("Score")
    plt.show()

    x_3=[1,2,3,4,5]
    aspects=[Reliability, Availability, Usability, Pertinence, Validity]
    labels_asp=['Reliability', 'Availability', 'Usability', 'Pertinence', 'Validity']
    plt.bar(x_3,aspects, tick_label=labels_asp, width=0.5)
    plt.xlabel("Quality Aspects")
    plt.ylabel("Score")
    plt.show()


    



    return dprof_df

import time
start = time.time()
cols2profile = df_taxi.columns
dprofile = dataprofile(df_taxi,cols2profile)
end = time.time()
#print('Time taken to execute dataprofile function ', (end - start)/60,' minutes')


dprofile
    