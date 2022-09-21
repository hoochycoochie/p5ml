import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import math
from sklearn.impute import SimpleImputer
from functions import utils
from functions import functions as utils2
from functools import reduce
from functions import utils
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder,StandardScaler
    




 
toys_and_game_gift={
    'name':'toys_and_game_gift',
    'data':[
    'baby',
    'toys','party_supplies',
    'cine_photo',
   
     'christmas_supplies'
]
}
arts_litterature={
    'name':'arts_litterature',
    'data':[
    'books_technical',
    'musical_instruments',
    'art',
         'music',
    'books_general_interest',
    'books_imported',
     
    'cds_dvds_musicals',
 'arts_and_craftmanship',
   
                 ]
}
 

electronics={
    'name':'electronics',
    'data':['electronics','computers_accessories','audio','dvds_blu_ray',
             'tablets_printing_image','telephony','fixed_telephony','computers','consoles_games']
}
foods_and_drinks={
    'name':'foods_and_drinks',
    'data':[
    'agro_industry_and_commerce','food','industry_commerce_and_business',
     'food_drink',
 'drinks',
   
     'la_cuisine'
]
}
house_ware_and_accessories={
    'name':"house_ware_and_accessories",
    'data':[
    'pet_shop','furniture_decor','garden_tools','housewares','bed_bath_table','office_furniture',
    'luggage_accessories','construction_tools_construction','construction_tools_lights','home_appliances',
    'kitchen_dining_laundry_garden_furniture','air_conditioning',
    'home_confort','small_appliances_home_oven_and_coffee',
    'signaling_and_security','small_appliances','costruction_tools_garden','home_construction',
    'construction_tools_safety',
    'furniture_living_room',
    'furniture_bedroom',
    'home_comfort_2',
    'furniture_mattress_and_upholstery','costruction_tools_tools',
    'security_and_services'   ,
        'home_appliances_2'
]
}
 
automobile={
    'name':'automobile',
    'data':['auto']
}

health_beauty={
    'name':'health_beauty',
    'data':['perfumery','health_beauty','diapers_and_hygiene','flowers']
}
fashion_and_cloth_shoes={
    'name':'fashion_and_cloth_shoes',
    'data':[
    'fashion_bags_accessories','watches_gifts',
    'fashion_underwear_beach','fashion_male_clothing',
    'fashion_shoes',
     'fashion_childrens_clothes',
        'fashio_female_clothing'
]
}
sports_and_accessories={
    'name':'sports_and_accessories',
    'data':['sports_leisure','fashion_sport']
}
 
others = {
    'name':'others',
    'data':['cool_stuff','stationery', 'market_place']
}
all_categories = [
    others,
    sports_and_accessories,
    fashion_and_cloth_shoes,
    health_beauty,
    automobile,
    house_ware_and_accessories,
    foods_and_drinks,
    electronics,
    arts_litterature,
    toys_and_game_gift
]

def create_df(from_date,to_date,select_customers=False,customers_uniques_ids=[]):
    customers_df=pd.read_csv('data/olist_customers_dataset.csv')
    geolocation_df=pd.read_csv('data/olist_geolocation_dataset.csv')
    order_items_df=pd.read_csv('data/olist_order_items_dataset.csv')
    order_payments_df=pd.read_csv('data/olist_order_payments_dataset.csv')
    order_reviews_df=pd.read_csv('data/olist_order_reviews_dataset.csv')
    orders_df=pd.read_csv('data/olist_orders_dataset.csv')
    products_df=pd.read_csv('data/olist_products_dataset.csv')
    products_df=products_df[~pd.isna(products_df['product_category_name'])]
    sellers_df=pd.read_csv('data/olist_sellers_dataset.csv')
    product_category_name_translation_df=pd.read_csv('data/product_category_name_translation.csv')


    cols_to_datetime=['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']
    for col in cols_to_datetime:
        orders_df[col] = orders_df[col].astype('datetime64[ns]')

    mask = (orders_df['order_purchase_timestamp'] >= from_date) & (orders_df['order_purchase_timestamp'] <= to_date)
    orders_df = orders_df.loc[mask]
    #print('len orders',len(orders_df))

    if select_customers:
        customers_df = customers_df.loc[customers_df['customer_unique_id'].isin(customers_uniques_ids)]

    #print('len customers_df',len(customers_df))
    customers=pd.merge(customers_df,geolocation_df,how='left',left_on='customer_zip_code_prefix',right_on='geolocation_zip_code_prefix',indicator=True)


    customers = customers.rename(columns={
        'geolocation_lat': 'customer_lat', 
        'geolocation_lng':'customer_lng',
        'geolocation_zip_code_prefix':'customer_zip_code',
        'geolocation_city':'customer_city',
        'geolocation_state':'customer_state'
    })

    #customers=customers[customers['_merge']=='left_only']


    if '_merge' in customers.columns.tolist():
        customers.drop('_merge',axis=1,inplace=True)


    ## transformation des variables review_creation_date,review_answer_timestamp d'object à datetime

 
    for col in ['review_creation_date','review_answer_timestamp']:
        order_reviews_df[col] = order_reviews_df[col].astype('datetime64[ns]')

    ## création des variables délais de réponse auite à une demande d'avis


    order_reviews_df['delay_answer_review']=(order_reviews_df['review_answer_timestamp'] - order_reviews_df['review_answer_timestamp']) / np.timedelta64(1, 'D')
    ## transformation de variable order_purchase_timestamp d'object à datetime
    cols_to_datetime=['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']
    for col in cols_to_datetime:
        orders_df[col] = orders_df[col].astype('datetime64[ns]')


        ## création des variables délais réels  de livraison, estimation délai de livraison

    orders_df['delay_delivery_time']=(orders_df['order_delivered_customer_date'] - orders_df['order_purchase_timestamp']) / np.timedelta64(1, 'D')
    ## création des variables heure, jour, et mois de l'achat

    orders_df["purchase_hour"] = orders_df["order_purchase_timestamp"].map(lambda d: d.hour)
    orders_df["purchase_weekday"] = orders_df["order_purchase_timestamp"].map(lambda d: d.day_name())
    orders_df["purchase_month"] = orders_df["order_purchase_timestamp"].map(lambda d: d.month)
    ##création de la variable volume
    products_df["product_volume_cm3"] = products_df["product_length_cm"] \
                               * products_df["product_height_cm"] \
                               * products_df["product_width_cm"]

    products=pd\
    .merge(
        products_df,
        product_category_name_translation_df,
        how='left',
        on='product_category_name',
        indicator=True
    )
    products = products.rename(columns={"product_category_name_english":
                                    "category"})
    products.drop('_merge',axis=1,inplace=True)
    orders = pd.merge(order_items_df,orders_df,how='left',on='order_id',indicator=True)
    orders.drop('_merge',axis=1,inplace=True)
    orders_customers = pd.merge(orders,customers_df,how='left',on='customer_id',indicator=True)
    

    if '_merge' in orders_customers.columns.tolist():
        orders_customers.drop('_merge',axis=1,inplace=True)

    orders_customers_products = pd.merge(orders_customers,products,how='left',on='product_id',indicator=True)
    if '_merge' in orders_customers_products.columns.tolist():
        orders_customers_products.drop('_merge',axis=1,inplace=True)
    #print('orders_customers_products.columns.tolist()',orders_customers_products.columns.tolist())
    orders_customers_products_reviews = pd.merge(
    orders_customers_products,order_reviews_df,how='left',on='order_id',indicator=True)
    #print('orders_customers_products_reviews.columns.tolist()',orders_customers_products_reviews.columns.tolist())
    order_payments_df_by_order=order_payments_df.groupby(by='order_id').agg({
        'payment_sequential':'count',
        'payment_installments':'sum'
        
    }).rename(columns={
    "payment_sequential": "count_payment_sequential",
    "payment_installments": "total_payment_installments"})
    orders_customers_products_reviews_payments = pd.merge(
    orders_customers_products_reviews,
        order_payments_df_by_order
        ,
        how='left',
        on='order_id'
    )

    if '_merge' in orders_customers_products_reviews_payments.columns.tolist():
        orders_customers_products_reviews_payments.drop('_merge',axis=1,inplace=True)

    


    orders_customers_products_reviews_payments_sellers = pd.merge(
        orders_customers_products_reviews_payments,sellers_df,how='left',on='seller_id',indicator=True)

    orders_customers_products_reviews_payments_sellers = orders_customers_products_reviews_payments_sellers[~pd.isna(orders_customers_products_reviews_payments_sellers.customer_unique_id)]
    #print('orders_customers_products_reviews_payments_sellers',orders_customers_products_reviews_payments_sellers.columns.tolist())
    ## variables comportant plus de 50% de valeurs manquantes
    na_df=utils.columns_na_percentage(orders_customers_products_reviews_payments_sellers)
    na_columns=na_df[na_df['na_rate_percent']>=50] ## 7 colonnes ont plus de 50% de valeurs nulles

    #print(na_columns['Column'])
    orders_customers_products_reviews_payments_sellers.drop(columns=na_columns['Column'].tolist(),axis=1,inplace=True)
        ## 1636 commandes dont la catégorie de produits est inconnue
    orders_to_remove=orders_customers_products_reviews_payments_sellers[pd.isna(orders_customers_products_reviews_payments_sellers.category)][['category']]
    df=orders_customers_products_reviews_payments_sellers.drop(index=orders_to_remove.index)


    all_aggregate_dfs=[]
   
    nb_orders_per_customer = df.groupby(by='customer_unique_id').\
    agg({'customer_id':'count'}).\
    rename(columns={'customer_id':'nb_orders'})

    all_aggregate_dfs.append(nb_orders_per_customer)

    nb_products_per_order_per_client=df.groupby(['customer_unique_id','order_id'])\
  .agg({'order_item_id':'count'})\
  .rename(columns={'order_item_id':'nb_products'})

    mean_nb_products_per_order_per_client=nb_products_per_order_per_client.\
    groupby(by='customer_unique_id').agg({'nb_products':'mean'}).\
    rename(columns={'nb_products':'mean_nb_products'})
    all_aggregate_dfs.append(mean_nb_products_per_order_per_client)
    total_price_freight_per_customer=df.groupby(by=['customer_unique_id']).\
    agg({'price':'sum','freight_value':'sum'}).\
    rename(columns={'price':'total_price_spent','freight_value':'total_freight_value_spent'})
    total_price_freight_per_customer['total_spent']=total_price_freight_per_customer['total_price_spent']+\
    total_price_freight_per_customer['total_freight_value_spent']
    all_aggregate_dfs.append(total_price_freight_per_customer)
    mean_delivery_delay_per_client=df.groupby(by='customer_unique_id').\
    agg({'delay_delivery_time':'mean'}).\
    rename(columns={'delay_delivery_time':'mean_delay_delivery_time'})
    all_aggregate_dfs.append(mean_delivery_delay_per_client)
    df[pd.isna(df['review_id'])][['review_score','review_id']]
    df['review_score']=np.where(pd.isna(df['review_id']),df['review_score'].mean(),df['review_score'])
    mean_score_review_per_customer=df.groupby(by='customer_unique_id').\
    agg({'review_score':'mean'}).\
    rename(columns={'review_score':'mean_review_score'})
    all_aggregate_dfs.append(mean_score_review_per_customer)
    mean_payment=df.groupby(by='customer_unique_id').\
    agg({'count_payment_sequential':'mean','total_payment_installments':'mean'}).\
    rename(columns={'count_payment_sequential':'mean_count_payment_sequential',
                    'total_payment_installments':'mean_total_payment_installments'})
    all_aggregate_dfs.append(mean_payment)

    favorite_month_per_customer=df.groupby(by='customer_unique_id').\
    agg({'purchase_month':lambda x:x.value_counts().index[0]}).\
    rename(columns={'purchase_month':'favorite_month'})
    all_aggregate_dfs.append(favorite_month_per_customer)
    favorite_category_per_customer=df.groupby(by='customer_unique_id').\
    agg({'category':lambda x:x.value_counts().index[0]}).\
    rename(columns={'category':'favorite_category'})
    #all_aggregate_dfs.append(favorite_category_per_customer)
    def add_main_category(x):
        for cat in all_categories:
        
            if x in cat['data']:
                return cat['name']

    favorite_category_per_customer['main_favorite_category']=favorite_category_per_customer['favorite_category'].\
    apply(add_main_category)
    all_aggregate_dfs.append(favorite_category_per_customer)

    last_score = df.groupby(by='customer_unique_id').\
    agg({'order_purchase_timestamp':lambda x:x.max(),
        # 'review_score':lambda x:x,
          'review_score':'mean',
         'review_id':lambda x:  x
        }).\
    rename(columns={
    'order_purchase_timestamp':'last_order_purchase_timestamp',
    'review_score':'last_review_score'
    
})

    last_score['has_reviewed_last']= np.where(pd.isna(last_score['review_id']), 0, 1)
    last_score=last_score.drop('review_id', axis=1)


    last_score['last_review_score']=pd.to_numeric(last_score['last_review_score'])
    cols_to_impute=[
        
        'last_review_score'
        
    ]
    #print('lastscore',last_score['last_review_score'])
    imputer = SimpleImputer(strategy='mean')
    if len(last_score['last_review_score'])>0:
        for col in cols_to_impute:
            data=np.array(last_score[col]).reshape((len(last_score[col]), 1))
            last_score[col]=imputer.fit_transform(data)
        all_aggregate_dfs.append(last_score)

    

    orders_by_regions=df.groupby(['customer_state']).\
    agg({'customer_id':'count'}).\
    rename(columns={'customer_id':'nb_orders_by_state'})
    state_most_orders = orders_by_regions.sort_values(ascending=False,by='nb_orders_by_state').head(1).index[0]
    is_from_most_orders_state = df[['customer_state','customer_unique_id']]
    is_from_most_orders_state['is_from_most_orders_state']=np.where( df['customer_state']==state_most_orders,1,0)



    all_aggregate_dfs.append(is_from_most_orders_state)
    df_customer = reduce(lambda df1,df2: pd.merge(df1,df2,on='customer_unique_id'), all_aggregate_dfs)

    for col in ['mean_count_payment_sequential_y', 'mean_total_payment_installments_y']:
        if col in df_customer.columns.tolist():
            df_customer.drop(col, axis=1, inplace=True)
    df_customer['last_review_score']=np.where(df_customer['has_reviewed_last']==0,0,df_customer['last_review_score'])
    cols_to_impute=[
    'mean_delay_delivery_time','mean_count_payment_sequential',
    'mean_total_payment_installments',
     
    
]
    imputer = SimpleImputer(strategy='mean')

    for col in cols_to_impute:
        data=data=np.array(df_customer[col]).reshape((len(df_customer[col]), 1))
        df_customer[col]=imputer.fit_transform(data)

  
    reference_date=to_date



    df_customer['last_order_delay']=(reference_date-df_customer['last_order_purchase_timestamp'] ) / np.timedelta64(1, 'D')
    df_customer.drop(index=df_customer[df_customer['customer_unique_id'].duplicated()].index, inplace=True)
    
    df_customer.isnull().sum()/len(df_customer)*100
    unrelevant_cols = ['favorite_category','last_order_purchase_timestamp']
    relevant_cols = [x for x in df_customer.columns.tolist() if x not in unrelevant_cols]
    df_customer_final = df_customer[relevant_cols]
    
   


    ohe = OneHotEncoder(sparse=False)

    
    data= ohe.fit_transform(
        np.array(df_customer_final['main_favorite_category']).reshape((len(df_customer_final['main_favorite_category']), 1))
        # df_final['main_favorite_category'].values.reshape((len(df_final['main_favorite_category']), 1)
    )
    new_cols= ohe.get_feature_names_out().tolist()
    df_hot= pd.DataFrame(columns =new_cols,
                data =data)
    
    for col in new_cols:
        df_customer_final[col]=df_hot[col]
        df_customer_final[col]=df_customer_final[col].replace(np.nan,0)


     
    cols_to_normalize= [
        'mean_nb_products',
        'nb_orders',
        'total_price_spent',
        'total_freight_value_spent',
        'mean_delay_delivery_time',
        'mean_review_score',
        'total_spent',
        'mean_delay_delivery_time',
        'mean_review_score',
        'mean_count_payment_sequential',
        'mean_total_payment_installments',
        #'total_freight_value_spent_percentage'
        
    ]

    df_customer_final[cols_to_normalize] =  StandardScaler().fit_transform(df_customer_final[cols_to_normalize].values)
    included_cols=[col for col in df_customer_final.columns.tolist() if col not in 
               
               ['customer_unique_id','customer_state','main_favorite_category']]
    data=df_customer_final[included_cols].values
    pca=PCA(0.95)
    pca.fit(data)
    columns = ['pca_%i' % i for i in range(pca.n_components_)]
    df_pca_cats = pd.DataFrame(pca.transform(data), columns=columns, index=df_customer_final.index)

    for col in columns:
        df_customer_final[col]=df_pca_cats[col]
    if select_customers==False:
        #df_customer_final.to_csv()
        df_customer_final.to_csv('data/customers_cleaned_2017.csv',index=False)
    return df_customer_final
 
    