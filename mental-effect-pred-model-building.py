import pandas as pd
dataset = pd.read_csv('mxmh_survey_results.csv')

data = dataset.drop(['Timestamp','Age','Primary streaming service','Permissions'], axis=1)
data2 = data.drop(['BPM'], axis=1)

ds = pd.DataFrame(data2)
df = ds.dropna()

df = df.copy()
target = 'Music effects'

# Specify the columns that want to keep
columns_to_keep = ['Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Exploratory', 'Foreign languages', 'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]', 'Frequency [Hip hop]', 'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]', 'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]', 'Anxiety', 'Depression', 'Insomnia', 'OCD','Music effects']

# Drop all columns except the ones specified
df = df[columns_to_keep]

encode = ['While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Exploratory', 'Foreign languages'
               , 'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]'
               , 'Frequency [Hip hop]', 'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]'
               , 'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]


target_mapper = {'Improve':0, 'No effect':1, 'Worsen':2}
def target_encode(val):
    return target_mapper[val]

df['Music effects'] = df['Music effects'].apply(target_encode)

# Separating X and y
x = df.drop('Music effects', axis=1)
y = df['Music effects']


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create an SVM classifier
svm = SVC(kernel='linear',random_state=0)

# Train the SVM classifier
svm.fit(x_train, y_train)


# Saving the model
import pickle
pickle.dump(svm, open('mental_effect_pred_svm.pkl', 'wb'))