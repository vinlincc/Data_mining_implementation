# setup
from pyspark import SparkContext
import json
import os
import sys
import time

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]', 'task22')
sc.setLogLevel("WARN")
file_folder_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]

test_rdd = sc.textFile(test_file_path)
header = test_rdd.first()
test_rdd = test_rdd.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))

yelp_train_rdd = sc.textFile(file_folder_path + '/yelp_train.csv')
header = yelp_train_rdd.first()
yelp_train_rdd = yelp_train_rdd.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0], (x[1], float(x[2])))).cache()

item2user_rdd = yelp_train_rdd.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().mapValues(list).cache()
item2user_dict = item2user_rdd.collectAsMap()
item2user_centered_rdd = item2user_rdd.mapValues(lambda x: {u: r - sum([pair[1] for pair in x]) / len(x) for u, r in x}).collectAsMap()
user2item = yelp_train_rdd.groupByKey().mapValues(list).collectAsMap()

def pearson_correlation(id1, id2):
    dict1 = item2user_centered_rdd.get(id1, {})
    dict2 = item2user_centered_rdd.get(id2, {})
    intersect = set(dict1.keys()) & set(dict2.keys())
    if len(intersect) < 3:
        return None
    
    nominator = sum(dict1[i] * dict2[i] for i in intersect)
    denominator = 0.00000000001 + (sum(dict1[i] ** 2 for i in intersect) ** 0.5) * (sum(dict2[i] ** 2 for i in intersect) ** 0.5)
    res = nominator / denominator
    if res < 0:
        res *= 0.1
    else:
        res *= 1.1
    return res

def predict(uid, iid):
    if uid not in user2item: return 3.75
    
    rated_items = user2item[uid]
    
    if iid not in item2user_centered_rdd: return sum([pair[1] for pair in rated_items]) / len(rated_items)
    
    correlations = []
    ratings = []
    for iid2, r in rated_items:
        c = pearson_correlation(iid, iid2)
        if c is not None:
            correlations.append(c)
            ratings.append(r)

    if len(correlations) >= 5:
        denominator = sum(abs(s) for s, _ in zip(correlations, ratings))
        nominator = sum(s * r for s, r in zip(correlations, ratings))
        if nominator < 30:
            item_ratings = [pair[1] for pair in item2user_dict.get(iid, [])]
            return sum(item_ratings) / len(item_ratings) if item_ratings else 3.75
        else:
            return nominator / denominator
    else:
        item_ratings = [pair[1] for pair in item2user_dict.get(iid, [])]
        return sum(item_ratings) / len(item_ratings) if item_ratings else 3.75


train_file_path = file_folder_path + '/review_train.json'

user_file_path = file_folder_path + '/user.json'
business_file_path = file_folder_path + '/business.json'
start = time.time()
train_rdd = sc.textFile(train_file_path).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id'], x['stars']))

user_rdd = sc.textFile(user_file_path).map(lambda x: json.loads(x))
business_rdd = sc.textFile(business_file_path).map(lambda x: json.loads(x))
user_rdd = user_rdd.map(lambda x: (x['user_id'], (x['review_count'], x['useful'], x['funny'], x['cool'], x['fans'], x['average_stars'], x['compliment_hot'], x['compliment_more'], x['compliment_profile'], x['compliment_cute'], x['compliment_list'], x['compliment_note'], x['compliment_plain'], x['compliment_cool'], x['compliment_funny'], x['compliment_writer'], x['compliment_photos'])))
business_rdd = sc.textFile(business_file_path).map(lambda x: json.loads(x))
business_rdd = business_rdd.map(lambda x: (
    x['business_id'], 
    (
        x['stars'], 
        x['review_count'], 
        x['is_open'], 
        x['attributes'].get('BikeParking', 'False') == 'True', 
        x['attributes'].get('BusinessAcceptsCreditCards', 'False') == 'True', 
        x['attributes'].get('GoodForKids','False') == 'True',  # Corrected
        x['attributes'].get('HasTV','False') == 'True',  # Corrected
        x['attributes'].get('NoiseLevel', 'null'),  # Corrected
        x['attributes'].get('OutdoorSeating','False') == 'True',  # Corrected
        x['attributes'].get('RestaurantsAttire','null'),  # Corrected
        x['attributes'].get('RestaurantsDelivery','False') == 'True',  # Corrected
        x['attributes'].get('RestaurantsGoodForGroups','False') == 'True',  # Corrected
        x['attributes'].get('RestaurantsPriceRange2','null'),  # Corrected
        x['attributes'].get('RestaurantsReservations','False') == 'True',  # Corrected
        x['attributes'].get('RestaurantsTakeOut','False') == 'True'  # Corrected
    )
) if x['attributes']
else (x['business_id'], (x['stars'], x['review_count'], x['is_open'], False, False, False, False, 'null', False, 'null', False, False, 'null', False, False))
)
def one_hot_encode_business_tuple(business_tuple):
    # Extract attributes from the tuple
    (business_id, (stars, review_count, is_open, has_bike_parking, accepts_credit_cards, 
                   has_garage, has_street_parking, noise_level, has_outdoor_seating, 
                   restaurants_attire, offers_delivery, good_for_groups, 
                   price_range, takes_reservations, has_takeout)) = business_tuple
    

    # Predefined unique categories for string fields
    noise_level_categories = ['quiet', 'average', 'loud', 'very_loud', 'null']
    attire_categories = ['casual', 'formal', 'dressy', 'null']
    price_range_categories = ['1', '2', '3', '4', 'null']
    
    # Generate one-hot vectors
    noise_level_vector = [int(noise_level == category) for category in noise_level_categories]
    attire_vector = [int(restaurants_attire == category) for category in attire_categories]
    price_range_vector = [int(price_range == category) for category in price_range_categories]
    
    # Combine all features into a new tuple, replacing string attributes with one-hot vectors
    transformed_tuple = (business_id, (stars, review_count, is_open, has_bike_parking, accepts_credit_cards, 
                                       has_garage, has_street_parking) + tuple(noise_level_vector) + 
                                       tuple(attire_vector) + (has_outdoor_seating, offers_delivery, 
                                       good_for_groups) + tuple(price_range_vector) + 
                                       (takes_reservations, has_takeout))
    
    return transformed_tuple
business_rdd = business_rdd.map(one_hot_encode_business_tuple)

# Join user features with ratings
user_train_rdd = train_rdd.map(lambda x: (x[0], (x[1], x[2]))) \
    .join(user_rdd) \
    .map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))  # Resulting structure: (business_id, (user_id, rating, user_features))

# Join business features
full_train_rdd = user_train_rdd.join(business_rdd) \
    .map(lambda x: (x[1][0][0], x[1][0][1], x[1][0][2], x[1][1]))  # Structure: (user_id, rating, user_features, business_features)

import numpy as np
import xgboost as xgb


train_local_data = full_train_rdd.collect()
labels = np.array([x[1] for x in train_local_data])
features = np.array([x[2] + x[3] for x in train_local_data])
xgb_reg = xgb.XGBRegressor(max_depth=5, eta=0.1, objective='reg:linear', eval_metric='rmse', n_estimators=100)


# Fit the model to the training data
xgb_reg.fit(features, labels)

business_feature_dict = business_rdd.collectAsMap()
user_feature_dict = user_rdd.collectAsMap()
test_data = test_rdd.collect()
test_features = []
for x in test_data:
    uid = x[0]
    bid = x[1]
    user_feature = (1,0,0,0,0,3.75,0,0,0,0,0,0,0,0,0,0,0)
    business_feature = (3.63,31.797,1,False,False,False,False,0,0,0,0,1,0,0,0,1,False,False,False,0,0,0,0,1,False,False)
    if uid in user_feature_dict: user_feature = user_feature_dict[uid]
    if bid in business_feature_dict: business_feature = business_feature_dict[bid]
    test_features.append(user_feature + business_feature)
test_features = np.array(test_features)
test_predictions = xgb_reg.predict(test_features)
test_predictions_cf = [predict(x[0], x[1]) for x in test_data]
test_predictions = [0.9 * p + 0.1 * cf for p, cf in zip(test_predictions, test_predictions_cf)]

res = "user_id, business_id, prediction\n"
for i in range(len(test_data)): 
    res += test_data[i][0] + "," + test_data[i][1] + ',' + str(test_predictions[i]) + "\n"
with open(output_file_path, "w") as f:
    f.writelines(res)
end = time.time()
print('Duration: ', end - start)