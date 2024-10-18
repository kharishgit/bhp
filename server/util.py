import json
import pickle
import numpy as np
__location = None
__data_column = None
__model = None

def get_estimated_prices(location,sqft,bhk,bath):
    try:
        loc_index = __data_column.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_column))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __location
def load_saved_artifacts():
    print("Loading saved artifacts")
    global __location
    global __data_column
    with open("artifacts/columns.json", 'r') as f:

        __data_column = json.load(f)['data_columns']
        __location = __data_column[3:]
    global __model
    with open("artifacts/bengaluru_House_DataModel.pickle" , 'rb') as f:
        __model = pickle.load(f)
    print("loading artifacts done")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_prices('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_prices('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_prices('Kalhalli', 1000, 2, 2))  # other location
    print(get_estimated_prices('Ejipura', 1000, 2, 2))  # other location

