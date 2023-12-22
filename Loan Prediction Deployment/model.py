import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
#Load the dataset
prosper_data =pd.read_csv("ProsperLoanDataset.csv")
y = prosper_data["LoanStatus"]
X = prosper_data.drop('LoanStatus',axis=1)
#Categorical data
categorical_data = X.select_dtypes(include=['object'])
categorical_data.head()
#labelEncoding
le = LabelEncoder()
categorical_data = categorical_data.apply(LabelEncoder().fit_transform)
categorical_data.head()
#numerical data
numerical_data = X.select_dtypes(include=['int','float'])
numerical_data.head()

#Scaling the data. Initializing the scaler
scaler = StandardScaler()

# columns transformed to a Numpy array, converted back to PD dataframe
numerical_data_rescaled = pd.DataFrame(scaler.fit_transform(numerical_data), 
                                    columns = numerical_data.columns, 
                                    index = numerical_data.index)

#Scaling the data. Initializing the scaler
scaler = StandardScaler()

# columns transformed to a Numpy array, converted back to PD dataframe
numerical_data_rescaled = pd.DataFrame(scaler.fit_transform(numerical_data), 
                                    columns = numerical_data.columns, 
                                    index = numerical_data.index)

numerical_data_rescaled.head()
#Adding the categorical features to the scaled numerical scaled one
scaled_data = pd.concat([categorical_data,numerical_data_rescaled],axis=1)

#Calculating mutual information scores
mutual_info = mutual_info_classif(scaled_data,y)
mutual_info

# Indexing and sorting the scores
mutual_info = pd.Series(mutual_info)
mutual_info.index = scaled_data.columns
# Models using Mutual Information
#selecting the features having mutual information score more than 0.078
high_mi_score_columns=[s for s in mutual_info.sort_values(ascending=False).index if mutual_info[s]>0.078]
X = scaled_data[high_mi_score_columns]
#print(X.head(5))
#Train and test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
#Initializing and fitting the model
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
#print("prediction =====",y_pred)

pickle.dump(regressor,open ('model.pkl','wb'))