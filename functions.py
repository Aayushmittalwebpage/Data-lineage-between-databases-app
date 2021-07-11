"""

Last Edited : 10-07-2021

@author : Aayush Mittal

"""



import pandas as pd
import numpy as np
import itertools

import re
import scipy.stats
from scipy.stats import *
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm 
import seaborn as sns
import pylab as py 
warnings.filterwarnings('ignore')

from numpy import mean
from numpy import std


from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy import stats
from scipy.stats import pearsonr

from fuzzywuzzy import fuzz, process


#these 2 list will be filled throughout the compare_two_table function and will be used in the end to create the lineage graph visualisation
list_lineage_source=[]
list_lineage_target=[]

#these 3 list will be filled throughout the compare_two_table function and will be used in the end to create the Tabular lineage report

attribute1=[]
attribute2=[]
metric=[]



def standarise(df, column,pct,pct_lower):
    sc = StandardScaler() 
    y = df[column][df[column].notnull()].to_list()
    y.sort()
    len_y = len(y)
    y = y[int(pct_lower * len_y):int(len_y * pct)]
    len_y = len(y)
    yy=([[x] for x in y])
    sc.fit(yy)
    y_std =sc.transform(yy)
    y_std = y_std.flatten()
    return y_std,len_y,y

# returns which distribution best fits the data attribute
def fit_distribution(df, column,pct,pct_lower):
    # Set up list of candidate distributions to use

    y_std,size,y_org = standarise(df,column,pct,pct_lower)
    dist_names = ['weibull_min','norm','weibull_max','beta',
                'invgauss','uniform','gamma','expon', 'pearson3','triang']

    chi_square_statistics = []
    # 11 bins
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions

    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
    

        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = round(sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency),0)
        chi_square_statistics.append(ss)


    #Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)
    
    #returns distribution with lowest chi sqaure or highest fitness
    return results.iloc[0,0]


# Function that calculates Jaccard similarity between 2 sets of data
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union



# Extracting name of dataset
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name



# preprocess all dates attributes into numerical value
def preprocess_date(df):
    import datetime
    df['today'] = datetime.date.today()
    
    filteredColumns_date = df.dtypes[df.dtypes == np.dtype('<M8[ns]') ]
    # list of columns whose data type is date
    listOfColumnNames_date = list(filteredColumns_date.index)
    
    for p in listOfColumnNames_date:
        
        df[p+'_num_days'] = (pd.to_datetime(df['today']) - pd.to_datetime(df[p])).dt.days
        df.drop([p], axis = 1, inplace=True)

    df.drop(['today'], axis = 1, inplace=True) 



def dates_compare(df1, df2, list_num):
    
    s1=0
    r1=0

    for p in list_num:
        
        #only columns ending with _num_days (dates column) will be processed
        x = re.search('_num_days$', p[0])
        y = re.search('_num_days$', p[1])

        #only if both columns ends with _num_days (dates column) then will be processed
        if(x and y):


            len1 = len(df1[p[0]])
            len2 = len(df2[p[1]])

            numpy_df1 = df1[p[0]].dropna()
            numpy_df2 = df2[p[1]].dropna()

            numpy_arr1=numpy_df1.to_numpy()
            numpy_arr2=numpy_df2.to_numpy()

            if(numpy_arr1.size == numpy_arr2.size):
                
                
                corr, _ = pearsonr(numpy_arr1, numpy_arr2)
                
                #this threshold of 0.8 can be changed according to use case 
                if(corr>0.8):
                    print('Pearsons correlation between {} and {} : {}'.format(p[0], p[1], corr))
                    attribute1.append(get_df_name(df1)+'__'+p[0])
                    attribute2.append(get_df_name(df2)+'__'+p[1])
                    metric.append(['Pearson Correlation: {}'.format(corr)])

                jacc_score = jaccard_similarity(numpy_arr1,numpy_arr2)
                
                #this threshold of 0.3 can be changed according to use case 
                if(jacc_score>0.3):
                    print("There is significant jaccard similarity b/w {} and {} : {} and there have equal number of values".format(p[0], p[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+p[0])
                    attribute2.append(get_df_name(df2)+'__'+p[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                    r1+=1


                d1 = fit_distribution(df1, p[0],0.99,0.01)
                d2 = fit_distribution(df2, p[1],0.99,0.01)

                m1 = mean(numpy_arr1)
                m2 = mean(numpy_arr2)

                st1 = std(numpy_arr1)
                st2 = std(numpy_arr2)

                if(max(m1,m2)!=0):
                    m = (abs(m1-m2)/max(m1,m2))*100

                if(max(st1,st2)!=0):
                    s = (abs(st1-st2)/max(st1,st2))*100

                try:
                    #checking if the difference in mean is less than 5% of the max mean and similarly for std.
                    if(m<5 and s<5 and d1==d2):
                        print("Both {} and {} have mean and standard deviation within 5% ".format(p[0], p[1]))
                        attribute1.append(get_df_name(df1)+'__'+p[0])
                        attribute2.append(get_df_name(df2)+'__'+p[1])
                        metric.append(['Mean and Std. are within 5% and both follow {} distribution'.format(d1)])

                except:
                    pass

                #append the list for graph representation
                if(d1==d2 and jacc_score>0.3):

                    str1=get_df_name(df1)+'__'+p[0]
                    str2=get_df_name(df2)+'__'+p[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)
                    
                if(corr>0.8 and d1==d2):
                    s1+=1
                        
                        
            #when length of attributes are different, here we wont calculate correlation
            else:
                jacc_score = jaccard_similarity(numpy_arr1,numpy_arr2)
                if(jacc_score>0.5):
                    print("There is significant jaccard similarity b/w {} and {} : {} and there have equal number of values".format(p[0], p[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+p[0])
                    attribute2.append(get_df_name(df2)+'__'+p[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                    
                    r1+=1


                d1 = fit_distribution(df1, p[0],0.99,0.01)
                d2 = fit_distribution(df2, p[1],0.99,0.01)

                m1 = mean(numpy_arr1)
                m2 = mean(numpy_arr2)

                st1 = std(numpy_arr1)
                st2 = std(numpy_arr2)

                if(max(m1,m2)!=0):
                    m = (abs(m1-m2)/max(m1,m2))*100

                if(max(st1,st2)!=0):
                    s = (abs(st1-st2)/max(st1,st2))*100

                try:
                    if(m<5 and s<5 and d1==d2):
                        print("Both {} and {} have mean and standard deviation within 5% ".format(p[0], p[1]))
                        attribute1.append(get_df_name(df1)+'__'+p[0])
                        attribute2.append(get_df_name(df2)+'__'+p[1])
                        metric.append(['Mean and Std. are within 5% and both follow {} distribution'.format(d1)])
                        
                        s1+=1


                except:
                    pass


                if(d1==d2 and jacc_score>0.5):

                    str1=get_df_name(df1)+'__'+p[0]
                    str2=get_df_name(df2)+'__'+p[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)
                    
                    
                
    return r1,s1

# compares 2 numerical attributes

def num_compare(df1, df2, list_num):

    s2=0
    r2=0

    for p in list_num:
        x = re.search('_num_days$', p[0])
        y = re.search('_num_days$', p[1])

        #dates
        if(x==None and y==None):


            len1 = len(df1[p[0]])
            len2 = len(df2[p[1]])

            len1uniq=df1[p[0]].nunique(dropna=False) #length of unique values
            len2uniq=df2[p[1]].nunique(dropna=False)

            numpy_df1_uniq  = df1[p[0]].unique()
            numpy_df2_uniq  = df2[p[1]].unique()
            
            
            percent_unique = (abs(len1uniq-len2uniq)/max(len1uniq,len2uniq))*100

            numpy_df1 = df1[p[0]].dropna()
            numpy_df2 = df2[p[1]].dropna()

            numpy_arr1=numpy_df1.to_numpy()
            numpy_arr2=numpy_df2.to_numpy()

            if(numpy_arr1.size == numpy_arr2.size):
                
                corr, _ = pearsonr(numpy_arr1, numpy_arr2)
                
                if(corr>0.8):
                    print('Pearsons correlation between {} and {} : {}'.format(p[0], p[1], corr))
                    attribute1.append(get_df_name(df1)+'__'+p[0])
                    attribute2.append(get_df_name(df2)+'__'+p[1])
                    metric.append(['Pearson Correlation: {}'.format(corr)])

                jacc_score = jaccard_similarity(numpy_arr1,numpy_arr2)
                if(jacc_score>0.3):
                    print("There is significant jaccard similarity b/w {} and {} : {} and there have equal number of values".format(p[0], p[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+p[0])
                    attribute2.append(get_df_name(df2)+'__'+p[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                    r2+=1


                d1 = fit_distribution(df1, p[0],0.99,0.01)
                d2 = fit_distribution(df2, p[1],0.99,0.01)

                m1 = mean(numpy_arr1)
                m2 = mean(numpy_arr2)

                st1 = std(numpy_arr1)
                st2 = std(numpy_arr2)

                if(max(m1,m2)!=0):
                    m = (abs(m1-m2)/max(m1,m2))*100

                if(max(st1,st2)!=0):
                    s = (abs(st1-st2)/max(st1,st2))*100

                try:
                    if(m<5 and s<5 and d1==d2):
                        print("Both {} and {} have mean and standard deviation within 5% ".format(p[0], p[1])) 
                        attribute1.append(get_df_name(df1)+'__'+p[0])
                        attribute2.append(get_df_name(df2)+'__'+p[1])
                        metric.append(['Mean and Std. are within 5% and both follow {} distribution'.format(d1)])
                        

                except:
                    pass


                if(d1==d2 and jacc_score>0.5):

                    str1=get_df_name(df1)+'__'+p[0]
                    str2=get_df_name(df2)+'__'+p[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)
                    
                if(corr>0.8 and d1==d2):
                    s2+=1
                            
                            
                            
            elif(len1!=len2 and len1uniq==len2uniq):
                            
                jacc_score = jaccard_similarity(numpy_arr1,numpy_arr2)
                if(jacc_score>0.5):
                    print("There is significant jaccard similarity b/w {} and {} : {} and there have equal number of values".format(p[0], p[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+p[0])
                    attribute2.append(get_df_name(df2)+'__'+p[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                            
                    r2+=1

                d1 = fit_distribution(df1, p[0],0.99,0.01)
                d2 = fit_distribution(df2, p[1],0.99,0.01)
                    

                m1 = mean(numpy_arr1)
                m2 = mean(numpy_arr2)

                st1 = std(numpy_arr1)
                st2 = std(numpy_arr2)

                if(max(m1,m2)!=0):
                    m = (abs(m1-m2)/max(m1,m2))*100

                if(max(st1,st2)!=0):
                    s = (abs(st1-st2)/max(st1,st2))*100

                try:
                    if(m<5 and s<5):
                        print("Both {} and {} have mean and standard deviation within 5% ".format(p[0], p[1]))
                        attribute1.append(get_df_name(df1)+'__'+p[0])
                        attribute2.append(get_df_name(df2)+'__'+p[1])
                        metric.append(['Mean and Std. are within 5% and both follow {} distribution'.format(d1)])
                        s2+=1


                except:
                    pass


                if(d1==d2 and jacc_score>0.5):

                    str1=get_df_name(df1)+'__'+p[0]
                    str2=get_df_name(df2)+'__'+p[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)
                    
                
                        
                        
            elif(percent_unique<90):
                            
                jacc_score = jaccard_similarity(numpy_arr1,numpy_arr2)
                if(jacc_score>0.5):
                    print("There is significant jaccard similarity b/w {} and {} : {} and there have equal number of values".format(p[0], p[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+p[0])
                    attribute2.append(get_df_name(df2)+'__'+p[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                            
                    r2+=1

                d1 = fit_distribution(df1, p[0],0.99,0.01)
                d2 = fit_distribution(df2, p[1],0.99,0.01)
                    

                m1 = mean(numpy_arr1)
                m2 = mean(numpy_arr2)

                st1 = std(numpy_arr1)
                st2 = std(numpy_arr2)

                if(max(m1,m2)!=0):
                    m = (abs(m1-m2)/max(m1,m2))*100

                if(max(st1,st2)!=0):
                    s = (abs(st1-st2)/max(st1,st2))*100

                try:
                    if(m<5 and s<5):
                        print("Both {} and {} have mean and standard deviation within 5% ".format(p[0], p[1]))
                        attribute1.append(get_df_name(df1)+'__'+p[0])
                        attribute2.append(get_df_name(df2)+'__'+p[1])
                        metric.append(['Mean and Std. are within 5% and both follow {} distribution'.format(d1)])
                        s2+=1


                except:
                    pass


                if(d1==d2 and jacc_score>0.5):

                    str1=get_df_name(df1)+'__'+p[0]
                    str2=get_df_name(df2)+'__'+p[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)
                    
                    
    return r2, s2

# compares 2 text attributes

def txt_compare(df1, df2, list_txt):

    r3=0
    s3=0

    for t in list_txt:
            len1 = len(df1[t[0]])
            len2 = len(df2[t[1]])

            len1uniq=df1[t[0]].nunique(dropna=False)
            len2uniq=df2[t[1]].nunique(dropna=False)

            numpy_df1_uniq  = df1[t[0]].unique()
            numpy_df2_uniq  = df2[t[1]].unique()
            

            numpy_arr1_uniq = numpy_df1_uniq.tolist()
            numpy_arr2_uniq = numpy_df2_uniq.tolist()


            numpy_df1 = df1[t[0]].to_numpy()
            numpy_df2 = df2[t[1]].to_numpy()


            numpy_arr1=numpy_df1.tolist()
            numpy_arr2=numpy_df2.tolist()

            percent_unique = (abs(len1uniq-len2uniq)/max(len1uniq,len2uniq))*100


            if(len1 == len2 and len1uniq!=len2uniq):

                jacc_score = jaccard_similarity(numpy_arr1,numpy_arr2)
                if(jacc_score>0.5):
                    print("There is significant jaccard similarity b/w {} and {} : {} and they are of same length".format(t[0], t[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+t[0])
                    attribute2.append(get_df_name(df2)+'__'+t[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                    r3+=1

                fuzz_ratio = fuzz.partial_ratio(numpy_arr1,numpy_arr2)
                if(fuzz_ratio>40):
                    print("There is significant Fuzzy ratio b/w {} and {} : {} and they are of same length".format(t[0], t[1], fuzz_ratio))
                    attribute1.append(get_df_name(df1)+'__'+t[0])
                    attribute2.append(get_df_name(df2)+'__'+t[1])
                    metric.append(['Significant fuzzy ratio: {}'.format(fuzz_ratio)])
                    s3+=1
                    
                if(jacc_score>0.5 or fuzz_ratio>40):
                    str1=get_df_name(df1)+'__'+t[0]
                    str2=get_df_name(df2)+'__'+t[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)
                    
            elif(len1 == len2 and len1uniq==len2uniq):

                jacc_score = jaccard_similarity(numpy_arr1,numpy_arr2)
                if(jacc_score>0.3):
                    print("There is significant jaccard similarity b/w {} and {} : {} and they are of same length and consist of equal number of unique value".format(t[0], t[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+t[0])
                    attribute2.append(get_df_name(df2)+'__'+t[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                    r3+=1

                fuzz_ratio = fuzz.partial_ratio(numpy_arr1,numpy_arr2)
                if(fuzz_ratio>40):
                    print("There is significant Fuzzy ratio b/w {} and {} : {} and they are of same length and consist of equal number unique value".format(t[0], t[1], fuzz_ratio))
                    attribute1.append(get_df_name(df1)+'__'+t[0])
                    attribute2.append(get_df_name(df2)+'__'+t[1])
                    metric.append(['Significant fuzzy ratio: {}'.format(fuzz_ratio)])
                    s3+=1
                    
                if(jacc_score>0.3 and fuzz_ratio>40):
                    str1=get_df_name(df1)+'__'+t[0]
                    str2=get_df_name(df2)+'__'+t[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)



            elif(len1 != len2 and len1uniq==len2uniq):

                jacc_score = jaccard_similarity(numpy_arr1_uniq,numpy_arr2_uniq)
                if(jacc_score>0.3):
                    print("There is significant jaccard similarity b/w unique value of {} and {} : {}".format(t[0], t[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+t[0])
                    attribute2.append(get_df_name(df2)+'__'+t[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                    r3+=1

                fuzz_ratio = fuzz.partial_ratio(numpy_arr1_uniq,numpy_arr2_uniq)
                if(fuzz_ratio>40):
                    print("There is significant Fuzzy ratio b/w {} and {} : {} and they are of same number of unique value".format(t[0], t[1], fuzz_ratio))
                    attribute1.append(get_df_name(df1)+'__'+t[0])
                    attribute2.append(get_df_name(df2)+'__'+t[1])
                    metric.append(['Significant fuzzy ratio: {}'.format(fuzz_ratio)])
                    s3+=1
                    
                if(jacc_score>0.3 and fuzz_ratio>40):
                    str1=get_df_name(df1)+'__'+t[0]
                    str2=get_df_name(df2)+'__'+t[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)

            elif(percent_unique<90):
                jacc_score = jaccard_similarity(numpy_arr1,numpy_arr2)
                if(jacc_score>0.3):
                    print("There is significant jaccard similarity b/w {} and {} : {}".format(t[0], t[1], jacc_score))
                    attribute1.append(get_df_name(df1)+'__'+t[0])
                    attribute2.append(get_df_name(df2)+'__'+t[1])
                    metric.append(['Significant Jaccard Similarity: {}'.format(jacc_score)])
                    r3+=1
                fuzz_ratio = fuzz.partial_ratio(numpy_arr1,numpy_arr2)
                if(fuzz_ratio>40):
                    print("There is significant Fuzzy ratio b/w {} and {} : {} ".format(t[0], t[1], fuzz_ratio))
                    attribute1.append(get_df_name(df1)+'__'+t[0])
                    attribute2.append(get_df_name(df2)+'__'+t[1])
                    metric.append(['Significant fuzzy ratio: {}'.format(fuzz_ratio)])
                    s3+=1
                    
                if(jacc_score>0.3 and fuzz_ratio>40):
                    str1=get_df_name(df1)+'__'+t[0]
                    str2=get_df_name(df2)+'__'+t[1]

                    list_lineage_source.append(str1)
                    list_lineage_target.append(str2)

    return r3, s3


# This function will take two datasets as argument and will do the following:
# 1) Make 2 list : a) list of combination of all numerical datatype and b) list of combination of text datatype
# 2) Then three functions will be called where all replication and similarity test will be carried for all combinations of attributes
# 3) Also the overall degree of similarity and replication will be printed in the end

def compare_two_tables(df1, df2):
    
    n1 = len(df1.columns)
    n2 = len(df2.columns)
    
    total_columns = n1+n2
    
    #df1
    filteredColumns_txt1 = df1.dtypes[df1.dtypes == np.object ]
    # list of columns whose data type is object string
    listOfColumnNames_txt1 = list(filteredColumns_txt1.index)

    filteredColumns_num1 = df1.dtypes[df1.dtypes == np.int64 ] + df1.dtypes[df1.dtypes == np.float64 ]
    # list of columns whose data type is object int or float
    listOfColumnNames_num1 = list(filteredColumns_num1.index)
    
    
    #df2
    filteredColumns_txt2 = df2.dtypes[df2.dtypes == np.object ]
    listOfColumnNames_txt2 = list(filteredColumns_txt2.index)

    filteredColumns_num2 = df2.dtypes[df2.dtypes == np.int64 ] + df2.dtypes[df2.dtypes == np.float64 ]
    listOfColumnNames_num2 = list(filteredColumns_num2.index)
    
    
    #combination of every couple of columns
    list_num = list(itertools.product(listOfColumnNames_num1, listOfColumnNames_num2))
    list_txt = list(itertools.product(listOfColumnNames_txt1, listOfColumnNames_txt2))
    
    r1, s1 = dates_compare(df1, df2, list_num)
    r2, s2 = num_compare(df1, df2, list_num)
    r3, s3 = txt_compare(df1, df2, list_txt)
    
    #overall degree of similarity between df1 and df2
    
    overall_degree_replication = (r1+r2+r3)/max(n1,n2)
    
    overall_degree_similarity = (s1+s2+s3)/max(n1,n2)
    
    print("Overall degree of replication between {} and {} is {}".format(get_df_name(df1), get_df_name(df2), overall_degree_replication))
    
    print("Overall degree of similarity between {} and {} is {}".format(get_df_name(df1), get_df_name(df2), overall_degree_similarity))

    return overall_degree_replication, overall_degree_similarity