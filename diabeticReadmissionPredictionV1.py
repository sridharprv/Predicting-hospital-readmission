
#%matplotlib inline
#import seaborn as sns
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('dataset/train.csv')
df.head()
df.info()
df = df.drop(['weight','tel_1', 'tel_2'],axis=1)
drop_Idx = set(df[(df['tel_9'] == '?') & (df['tel_10'] == '?') & (df['tel_11'] == '?')].index)

drop_Idx = drop_Idx.union(set(df['tel_9'][df['tel_9'] == '?'].index))
drop_Idx = drop_Idx.union(set(df['tel_10'][df['tel_10'] == '?'].index))
drop_Idx = drop_Idx.union(set(df['tel_11'][df['tel_11'] == '?'].index))
drop_Idx = drop_Idx.union(set(df['race'][df['race'] == '?'].index))
drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 11].index))
drop_Idx = drop_Idx.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))
new_Idx = list(set(df.index) - set(drop_Idx))
df = df.iloc[new_Idx]
df = df.drop(['tel_30', 'tel_41','tel_47','tel_20','tel_28','tel_29','tel_45','tel_46','tel_47'], axis = 1)
df['gender'] = df['gender'].apply(lambda x: 0 if x == "Female" else 1)
def agecategory(x):
    
    if x == "[0-10)" :
        return 5
    elif x == "[10-20)":
        return 15
    elif x == "[20-30)":
        return 25
    elif x == "[30-40)":
        return 35
    elif x == "[40-50)":
        return 45
    elif x == "[50-60)":
        return 55
    elif x == "[60-70)":
        return 65
    elif x == "[70-80)":
        return 75
    else:
        return 0
df['age'] = df['age'].apply(lambda x: agecategory(x))
indicators = ['tel_15', 'tel_16', 'tel_17', 'tel_18', 'tel_19', 'tel_21', 'tel_22',
       'tel_23', 'tel_24', 'tel_25', 'tel_26', 'tel_27', 'tel_42', 'tel_43',
       'tel_44']
for i in indicators:
    df[i] = df[i].apply(lambda x: 0 if x == "No" else 1)
df['total_indicators'] = np.zeros((len(df['tel_15'])))
for col in indicators:
    df['total_indicators'] += df[col]
df['tel_13'] = df['tel_13'].apply(lambda x: 0 if x == "None" else (1 if x=="Norm" else 2) )
df['tel_14'] = df['tel_14'].apply(lambda x: 0 if x == "None" else (1 if x=="Norm" else 2) )
#patients = df['patient_id']
#df[patients.isin(patients[patients.duplicated()])]
#df = df.drop_duplicates(subset= ['patient_id'], keep = 'first')
df['admission_type_id'] = df['admission_type_id'].astype('object')
df['admission_source_id'] = df['admission_source_id'].astype('object')
df['discharge_disposition_id'] = df['discharge_disposition_id'].astype('object')
df['tel_13'] = df['tel_13'].astype('object')
df['tel_14'] = df['tel_14'].astype('object')
delete_columns =['encounter_id','admission_type_id','discharge_disposition_id','admission_source_id', 'tel_3', 'tel_4', 'tel_5', 'tel_6', 'tel_7',
       'tel_8', 'tel_9', 'tel_10', 'tel_11', 'tel_15', 'tel_16', 'tel_17', 'tel_18', 'tel_19', 'tel_21', 'tel_22',
       'tel_23', 'tel_24', 'tel_25', 'tel_26', 'tel_27', 'tel_42', 'tel_43',
       'tel_44', 'tel_49']
df.drop(delete_columns, inplace=True, axis=1)
df['tel_48'] = df['tel_48'].apply(lambda x: 0 if x == "No" else 1)
df['tel_48'] = df['tel_48'].astype('object')

categorical=df.select_dtypes(include=['object'])
numeric=df.select_dtypes(exclude=['object'])
nominal_columns = ['race', 'tel_13', 'tel_14', 'tel_48']
dummy_df = pd.get_dummies(df[nominal_columns])
df = pd.concat([df, dummy_df], axis=1)
df = df.drop(nominal_columns, axis=1)
df1 = df
y = df1['diabetesMed']
X = df.drop('diabetesMed',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=64, max_features='sqrt')
clf = clf.fit(X_train, Y_train)
delete_features= ['gender', 'race_AfricanAmerican', 'race_Asian',
       'race_Caucasian', 'race_Hispanic', 'race_Other','tel_13_0', 'tel_13_1',
       'tel_13_2', 'tel_14_0', 'tel_14_1', 'tel_14_2']
df_final = df1.drop(delete_features, axis = 1)
y = df_final['diabetesMed']
X = df_final.drop('diabetesMed',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.20, random_state=42)

# DecisionTreeClassifier

performance = []
for max_depth in [2,3,5,7,10]:
    dTree = DecisionTreeClassifier(criterion='entropy', class_weight = "balanced", max_depth=max_depth)
    performance.append((max_depth, np.mean(cross_val_score(dTree, X_train, Y_train, cv = 10, scoring = "f1_micro"))))
from sklearn.model_selection import cross_val_score, KFold

dTree = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_depth = 5)
kf = KFold(n_splits=10, shuffle=True, random_state=0)
dTree.fit(X_train, Y_train)
y_prediction = dTree.predict(X_test)

scaler = StandardScaler()
X_train_normal = scaler.fit_transform(X_train)
X_test_normal = scaler.transform(X_test)

model = LogisticRegressionCV(Cs = 10, cv = 10, class_weight = "balanced")
model.fit(X_train_normal, Y_train)
y_prediction = model.predict(X_test_normal)

#test dataset
dframe = pd.read_csv("dataset/test.csv",header=0,skiprows=0,engine='python')
dframeclone = pd.read_csv("dataset/test.csv",header=0,skiprows=0,engine='python')
ids = dframe['encounter_id']
dframe[ids.isin(ids[ids.duplicated()])]
dframe = dframe.replace('?', np.NaN )
dframe = dframe.replace('Unknown/Invalid', np.NaN )
dframe.drop(['weight','tel_1', 'tel_2'],axis=1,inplace=True)
dframe.drop(['tel_30', 'tel_41','tel_47','tel_20','tel_28','tel_29','tel_45','tel_46','tel_47'], axis = 1,inplace=True)
drop_Idx = set(dframe[(dframe['tel_9'] == '?') & (dframe['tel_10'] == '?') & (dframe['tel_11'] == '?')].index)

drop_Idx = drop_Idx.union(set(dframe['tel_9'][dframe['tel_9'] == '?'].index))
drop_Idx = drop_Idx.union(set(dframe['tel_10'][dframe['tel_10'] == '?'].index))
drop_Idx = drop_Idx.union(set(dframe['tel_11'][dframe['tel_11'] == '?'].index))
drop_Idx = drop_Idx.union(set(dframe['race'][dframe['race'] == '?'].index))
drop_Idx = drop_Idx.union(set(dframe[dframe['discharge_disposition_id'] == 11].index))
drop_Idx = drop_Idx.union(set(dframe['gender'][dframe['gender'] == 'Unknown/Invalid'].index))
new_Idx = list(set(dframe.index) - set(drop_Idx))
dframe = dframe.iloc[new_Idx]
dframe['gender'] = dframe['gender'].apply(lambda x: 0 if x == "Female" else 1) 


dframe['age'] = dframe['age'].apply(lambda x: agecategory(x))
indicators = ['tel_15', 'tel_16', 'tel_17', 'tel_18', 'tel_19', 'tel_21', 'tel_22',
       'tel_23', 'tel_24', 'tel_25', 'tel_26', 'tel_27', 'tel_42', 'tel_43',
       'tel_44']
for i in indicators:
    dframe[i] = dframe[i].apply(lambda x: 0 if x == "No" else 1)
dframe['total_indicators'] = np.zeros((len(dframe['tel_15'])))
for col in indicators:
    dframe['total_indicators'] += dframe[col]
dframe['tel_13'] = dframe['tel_13'].apply(lambda x: 0 if x == "None" else (1 if x=="Norm" else 2) )
dframe['tel_14'] = dframe['tel_14'].apply(lambda x: 0 if x == "None" else (1 if x=="Norm" else 2) )
dframe['tel_48'] = dframe['tel_48'].apply(lambda x: 0 if x == "No" else 1) 


dframe['admission_type_id'] = dframe['admission_type_id'].astype('object')
dframe['admission_source_id'] = dframe['admission_source_id'].astype('object')
dframe['discharge_disposition_id'] = dframe['discharge_disposition_id'].astype('object')
dframe['tel_48'] = dframe['tel_48'].astype('object')
dframe['tel_13'] = dframe['tel_13'].astype('object')
dframe['tel_14'] = dframe['tel_14'].astype('object')
delete_columns =['admission_type_id','discharge_disposition_id','admission_source_id', 'tel_3', 'tel_4', 'tel_5', 'tel_6', 'tel_7','tel_8', 'tel_9', 'tel_10', 'tel_11', 'tel_15', 'tel_16', 'tel_17', 'tel_18', 'tel_19', 'tel_21', 'tel_22',       'tel_23', 'tel_24', 'tel_25', 'tel_26', 'tel_27', 'tel_42', 'tel_43',       'tel_44', 'tel_49']
dframe.drop(delete_columns, inplace=True, axis=1)
categorical=dframe.select_dtypes(include=['object'])
numeric=dframe.select_dtypes(exclude=['object'])

nominal_columns = ['race', 'tel_13', 'tel_14', 'tel_48']
dummy_dframe = pd.get_dummies(dframe[nominal_columns])
dframe = pd.concat([dframe, dummy_dframe], axis=1)
dframe = dframe.drop(nominal_columns, axis=1)
delete_features= ['race_AfricanAmerican', 'race_Asian',
       'race_Caucasian', 'race_Hispanic', 'race_Other','tel_13_0', 'tel_13_1',
       'tel_13_2', 'tel_14_0', 'tel_14_1', 'tel_14_2']
dframe = dframe.drop(delete_features, axis = 1)
dfnew= dframe.iloc[:,1:8]
y_prediction_DTC = dTree.predict(dfnew[1:])
dfnew = dfnew.reset_index(drop=True)
data2_scalar = scaler.fit_transform(dfnew)
y_prediction_LRC = model.predict(data2_scalar)
predicted_probability = model.predict_proba(data2_scalar)
predicted_probability = pd.DataFrame(predicted_probability)
dframe['y_prediction_LRC'] = pd.DataFrame(y_prediction_LRC)
dframe['y_prediction_DTC'] = pd.DataFrame(y_prediction_DTC)[0]
dframe['predicted_probability_LR_0'] = pd.DataFrame(predicted_probability)[0]
dframe['predicted_probability_LR_1'] = pd.DataFrame(predicted_probability)[1]
predicted_probability_decisiontree = dTree.predict_proba(dfnew)
dframe['predicted_probability_decisiontree_0'] = pd.DataFrame(predicted_probability_decisiontree)[0]
dframe['predicted_probability_decisiontree_1'] = pd.DataFrame(predicted_probability_decisiontree)[1]
prediction=dframe[['encounter_id','y_prediction_LRC']]
prediction.rename(columns={'y_prediction_LRC':'diabetesMed'},inplace=True) 
submit =  pd.merge(dframeclone, prediction, how='left', on=['encounter_id'])
submit['diabetesMed'].fillna(1,inplace=True)
submission = submit[['encounter_id','diabetesMed']] 
submission.to_csv('submission.csv',index=False)
print('submission file is generated')