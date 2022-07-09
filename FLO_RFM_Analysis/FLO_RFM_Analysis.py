#####################################
# Customer Segmentation with RFM
#####################################

###################
# Business Problem
###################
"""
FLO, an online shoe store, wants to segment its customers and determine marketing strategies
according to these segments.
For this purpose, the behaviors of the customers will be defined and the groups will be formed
according to the clusters.
"""

###################
# Features
###################

# Total Features   : 12
# Total Row        : 19.945
# csv File Size    : 2.7MB

"""
- master_id                         : Unique Customer Number
- order_channel                     : Which channel of the shopping platform is used 
                                      (Android, IOS, Desktop, Mobile)
- last_order_channel                : The channel where the most recent purchase was made
- first_order_date                  : Date of the customer's first purchase
- last_order_channel                : Customer's previous shopping history
- last_order_date_offline           : The date of the last purchase made by the customer on the offline platform
- order_num_total_ever_online       : Total number of purchases made by the customer on the online platform
- order_num_total_ever_offline      : Total number of purchases made by the customer on the offline platform
- customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
- customer_value_total_ever_online  :  Total fees paid for the customer's online purchases
- interested_in_categories_12       : List of categories the customer has shopped in the last 12 months
"""

# ----------------------------------------------------------------------------------------------------------------

import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


df["order_channel"].value_counts()

"""
Android App    9495
Mobile         4882
Ios App        2833
Desktop        2735
"""


df["interested_in_categories_12"].head()

"""
0                             [KADIN]
1    [ERKEK, COCUK, KADIN, AKTIFSPOR]
2                      [ERKEK, KADIN]
3                 [AKTIFCOCUK, COCUK]
4                         [AKTIFSPOR]
"""


# Omnichannel means that customers shop from both online and offline platforms.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# We convert the column types from object to datetime format.

# first_order_date                   object ---------datetime
# last_order_date                    object ---------datetime
# last_order_date_online             object ---------datetime
# last_order_date_offline            object ---------datetime

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()


# Distribution of the number of customers in shopping channels, total number of products purchased
# and total expenditures

df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})

"""
               master_id  order_num_total  customer_value_total
order_channel                                                  
Android App         9495        52269.000           7819062.760
Desktop             2735        10920.000           1610321.460
Ios App             2833        15351.000           2525999.930
Mobile              4882        21679.000           3028183.160
"""

# Top 10 customers with the most profits

df.groupby("master_id").agg({"customer_value_total": "sum"
                             }).sort_values(by="customer_value_total", ascending=False).head(10)

"""
                                customer_value_total
master_id                                                 
5d1c****-****-****-****             45905.100
d5ef****-****-****-****             36818.290
73fd****-****-****-****             33918.100
7137****-****-****-****             31227.410
47a6****-****-****-****             20706.340
a4d5****-****-****-****             18443.570
d696****-****-****-****             16918.570
fef5****-****-****-****             12726.100
"""

# Top 10 customers with the most orders

df.groupby("master_id").agg({"order_num_total": "sum"
                             }).sort_values(by="order_num_total", ascending=False).head(10)

"""
                              order_num_total
master_id                                            
5d1c****-****-****-****           202.000
cba5****-****-****-****           131.000
a57f****-****-****-****           111.000
fdbe****-****-****-****            88.000
3299****-****-****-****            83.000
73fd****-****-****-****            82.000
44d0****-****-****-****            77.000
b27e****-****-****-****            75.000
d696****-****-****-****            70.000
a4d5****-****-****-****            70.000
"""


# Calculating RFM Metrics

df["last_order_date"].max()   # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

rfm.head()

"""
               customer_id   recency   frequency    monetary
0  cc29****-****-****-****    95.000       5.000     939.370
1  f431****-****-****-****   105.000      21.000    2013.550
2  69b6****-****-****-****   186.000       5.000     585.320
3  1854****-****-****-****   135.000       2.000     121.970
4  d6ea****-****-****-****    86.000       2.000     209.980
"""

# Calculating RFM Scores

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.head()

rfm[rfm["RF_SCORE"] == "11"].head()

rfm[rfm["RF_SCORE"] == "55"].head()


# Definition of RF Scores as Segments

# Regex
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()


# The recency, frequency and monetary averages of the segments

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

"""
                    recency       frequency       monetary      
                       mean count      mean count     mean count
segment                                                         
about_to_sleep      113.785  1629     2.401  1629  359.009  1629
at_Risk             241.607  3131     4.472  3131  646.610  3131
cant_loose          235.444  1200    10.698  1200 1474.468  1200
champions            17.107  1932     8.934  1932 1406.625  1932
hibernating         247.950  3604     2.394  3604  366.267  3604
loyal_customers      82.595  3361     8.375  3361 1216.819  3361
need_attention      113.829   823     3.728   823  562.143   823
new_customers        17.918   680     2.000   680  339.956   680
potential_loyalists  37.156  2938     3.304  2938  533.184  2938
promising            58.921   647     2.000   647  335.673   647
"""


segments = rfm["segment"].value_counts().sort_values(ascending=False)

segments

"""
hibernating            3604
loyal_customers        3361
at_Risk                3131
potential_loyalists    2938
champions              1932
about_to_sleep         1629
cant_loose             1200
need_attention          823
new_customers           680
promising               647
"""

# Case Example

"""
A new brand of women's shoes will be included. The product prices of the brand to be included are 
above the general customer preferences. For this reason, customers in the profile who will be interested in
the promotion of the brand and product sales are requested to be contacted privately.Target customers 
(champions, loyal_customers) and shoppers from the female category. We need access to the ID numbers 
of these customers.
"""

segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
customer_ids = df[(df["master_id"].isin(segments_customer_ids)) &(df["interested_in_categories_12"].
                                                                      str.contains("KADIN"))]["master_id"]
customer_ids .head()

# Turn the csv format
customer_ids.to_csv("target_customers.csv", index=False)

