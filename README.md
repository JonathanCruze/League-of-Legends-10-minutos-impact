# Context
League of Legends is a MOBA (multiplayer online battle arena) where 2 teams (blue and red) face off. There are 3 lanes, a jungle, and 5 roles. The goal is to take down the enemy Nexus before the enemy destroys yours to win the game.

## Content
This dataset contains the first 10min. stats of approx. 10k ranked games (SOLO QUEUE) from a high ELO (DIAMOND I to MASTER). Players have roughly the same level.

Each game is unique. The gameId can help you to fetch more attributes from the Riot API.

There are 19 features per team (38 in total) collected after 10min in-game. This includes kills, deaths, gold, experience, level… It's up to you to do some feature engineering to get more insights.

The column blueWins is the target value (the value we are trying to predict). A value of 1 means the blue team has won. 0 otherwise.

*There is no missing values in the data set.*

# Glossary
`Warding totem:` An item that a player can put on the map to reveal the nearby area. Very useful for map/objectives control.

`Minions:` NPC that belong to both teams. They give gold when killed by players.

`Jungle minions:` NPC that belong to NO TEAM. They give gold and buffs when killed by players.

`Elite monsters:` Monsters with high hp/damage that give a massive bonus (gold/XP/stats) when killed by a team.

`Dragons:` Elite monster which gives team bonus when killed. The 4th dragon killed by a team gives a massive stats bonus. The 5th dragon *(Elder Dragon)* offers a huge advantage to the team.

`Herald:` Elite monster which gives stats bonus when killed by the player. It helps to push a lane and destroys structures.

`Towers:` Structures you have to destroy to reach the enemy Nexus. They give gold.

`Level:` Champion level. Start at 1. Max is 18.


# CLASSIFYING LOL HIGH-ELO RANKED GAMES BY LOOKING AT THE FIRST 10 MINUTES WORTH OF DATA 
This is my Logisitic Regression aproach to predicting the result of League of Legends first 10 minutes matches 

This model works by using `Logistic Regressions` in Python `Python 3.7.13`.

## Importing libraries
```bash
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
```

## Gettin' Started

Loading Data:
```python
# get pandas dataframe
data = pd.read_csv('LoL_10min_stats.csv')
data
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/3abb00f40e1a20493d74d9f3a43db29afc5bb41d/Screenshots/Initial_Pandas_dataFrame.png)

Viewing all Tag's:
```
data.info()

  RangeIndex: 9879 entries, 0 to 9878
  Data columns (total 40 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   gameId                        9879 non-null   int64  
     1   blueWins                      9879 non-null   int64  
     2   blueWardsPlaced               9879 non-null   int64  
     3   blueWardsDestroyed            9879 non-null   int64  
     4   blueFirstBlood                9879 non-null   int64  
     5   blueKills                     9879 non-null   int64  
     6   blueDeaths                    9879 non-null   int64  
     7   blueAssists                   9879 non-null   int64  
     8   blueEliteMonsters             9879 non-null   int64  
     9   blueDragons                   9879 non-null   int64  
     10  blueHeralds                   9879 non-null   int64  
     11  blueTowersDestroyed           9879 non-null   int64  
     12  blueTotalGold                 9879 non-null   int64  
     13  blueAvgLevel                  9879 non-null   float64
     14  blueTotalExperience           9879 non-null   int64  
     15  blueTotalMinionsKilled        9879 non-null   int64  
     16  blueTotalJungleMinionsKilled  9879 non-null   int64  
     17  blueGoldDiff                  9879 non-null   int64  
     18  blueExperienceDiff            9879 non-null   int64  
     19  blueCSPerMin                  9879 non-null   float64
     20  blueGoldPerMin                9879 non-null   float64
     21  redWardsPlaced                9879 non-null   int64  
     22  redWardsDestroyed             9879 non-null   int64  
     23  redFirstBlood                 9879 non-null   int64  
     24  redKills                      9879 non-null   int64  
     25  redDeaths                     9879 non-null   int64  
     26  redAssists                    9879 non-null   int64  
     27  redEliteMonsters              9879 non-null   int64  
     28  redDragons                    9879 non-null   int64  
     29  redHeralds                    9879 non-null   int64  
     30  redTowersDestroyed            9879 non-null   int64  
     31  redTotalGold                  9879 non-null   int64  
     32  redAvgLevel                   9879 non-null   float64
     33  redTotalExperience            9879 non-null   int64  
     34  redTotalMinionsKilled         9879 non-null   int64  
     35  redTotalJungleMinionsKilled   9879 non-null   int64  
     36  redGoldDiff                   9879 non-null   int64  
     37  redExperienceDiff             9879 non-null   int64  
     38  redCSPerMin                   9879 non-null   float64
     39  redGoldPerMin                 9879 non-null   float64
  dtypes: float64(6), int64(34)
```
With this, you can see that there is no NaN values in the data set, becouse all the columns have the same 9879 data size. 
```python
data.shape
```
(9879, 40)

## A short data analysis
Histogram for every tag in the data frame:
```python
data.hist(figsize=(20,20), **{"align": "mid"})
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/3abb00f40e1a20493d74d9f3a43db29afc5bb41d/Screenshots/Hist_Plots.png)

Correlated values:
```python
sns.clustermap(data.corr())
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/ba67aea0e6cfebf7fb0d6f590223c7f04d9f6510/Screenshots/Corr_Data_Plot.png)

With this correlational plot we were able to observe that some pairs of columns has a perfect correlation, for example: (blueKills, redDeaths). This is because these columns actually represent the same information, as each Blue Team Kill is obviusly a Red Team Death. 

These suggest that some columns can be removed later on.

## Separating the data by teams so i can look foword for further conclusions
Looking for the amount and percentage of games won by each team.
```python
blueCount = (data['blueWins'] == 1).sum()
redCount = (data['blueWins'] == 0).sum()

print('Number of times the BLUE team wins: {blueCount}'.format(blueCount = (data['blueWins'] == 1).sum()))
print('Number of times the RED team wins: {redCount}'.format(redCount = (data['blueWins'] == 0).sum()))
```
Number of times the BLUE team wins: *4930*
Number of times the RED team wins: *4949*

```python
# Data to plot
labels = ['Blue Team wins', 'Red Team Wins']
colors = ['lightskyblue', 'lightcoral']
slices = [blueCount, redCount]

# Plot
plt.pie(slices, labels = labels, colors = colors, 
autopct='%1.1f%%', shadow=True, wedgeprops = {'edgecolor': 'black'}, startangle=85)

plt.title('Overall Win Rate % per Team [Diamond-Master]')
plt.axis('equal')
plt.tight_layout()
plt.show()
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/3abb00f40e1a20493d74d9f3a43db29afc5bb41d/Screenshots/Pie_chart_WinRate.png)
 - Both teams have the basically the same winning percentage and it just differs by a little.

Looking if there are matches in which a team is winning and then losses.
```python
# First for blue threw games
sns.countplot(data=data[data['blueTotalGold'] > 20000], x='blueWins')
plt.xlabel('Red Comeback');
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/0f08521b5d28ed1c97ea0a3d18d658a48e6423ba/Screenshots/Red_Comeback_Plot.png)
```python
# Second for red threw games
sns.countplot(data=data[data['redTotalGold'] > 20000], x='blueWins')
plt.xlabel('Blue ComeBack');
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/0f08521b5d28ed1c97ea0a3d18d658a48e6423ba/Screenshots/Blue_Comeback_Plot.png)
- Apparently Red Team throws a little bit more than Blue

## Cleaning data
```python
#GameID isn't important, so i decided to remove it
data = data.drop('gameId', axis=1)

# We drop all the variables that are symetric i.e blueKills == redDeaths
symetric_features = ['redFirstBlood', 'blueDeaths', 'redDeaths', 'redGoldDiff', 'redExperienceDiff']

# We drop all the redundant variables. For example, we do not care about the total amount of gold earned, what matters is actually the difference compared to the enemy team
redundant_features = ['redTotalGold', 'blueTotalGold', 'redAvgLevel', 'blueAvgLevel', 'redTotalExperience', 'blueTotalExperience', 'redTotalMinionsKilled','blueTotalMinionsKilled', 'blueKills', 'redKills']

# We drop highly correlated features. For example, the redAvgLevel is logically highly corelated with the 
multicolinear_features = ['redCSPerMin', 'blueCSPerMin', 'blueEliteMonsters', 'redEliteMonsters', 'blueGoldPerMin', 'redGoldPerMin']

# Kill difference
data = data.assign(blueKillDiff = data['blueKills'] - data['redKills'])
data = data.assign(redKillDiff = data['redKills'] - data['blueKills'])

# Remaining wards
data = data.assign(blueWardsRemain = data['blueWardsPlaced'] - data['redWardsDestroyed'])
data = data.assign(redWardsRemain = data['redWardsPlaced'] - data['blueWardsDestroyed'])
data = data.drop(['redWardsPlaced','redWardsDestroyed','blueWardsPlaced','blueWardsDestroyed'], axis = 1)

# Droppin features
data = data.drop(symetric_features, axis = 1)
data = data.drop(redundant_features, axis = 1)
data = data.drop(multicolinear_features, axis = 1)
```
Looking for high and low impact on determinating the Win condition 
```python
data.corr()['blueWins'].drop('blueWins').sort_values().plot(kind = 'bar')
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/145eeb0749d60509bc8816246e4f56f61d480c46/Screenshots/Corr_BlueWins_Plot.png)
```python
data.shape
```
```python
OUTPUT:
  (9879, 18)
```

```python
# Cleaned DataFrame
data
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/41141a6ddb05253ab184f5dc722526756c9b7300/Screenshots/Cleaned_Pandas_dataFrame.png)

## Blue and Red Team Data Split
Sepparating the data just for the Blue Team Tags:
```python
data_split = data

# Blue Team Data
blue_data_columns = []
for col in data_split.columns:
       if 'blue' in col:
              blue_data_columns.append(col)
```
```python
data_blue = data_split[blue_data_columns]
data_blue.head(7)
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/41141a6ddb05253ab184f5dc722526756c9b7300/Screenshots/BlueData_Pandas_dataFrame.png)

Sepparating the data just for the Red Team Tags:
```python
#Red Team Data
red_data_columns = []
for col in data_split.columns:
       if 'red' in col:
              red_data_columns.append(col)
```
```python
# Creating 'RedWins' tag
data_red = data_split[red_data_columns]
data_red['redWins'] = data_split['blueWins'].map({1:0,0:1})

# Reorganizing 'redWins' column
first_column = data_red.pop('redWins')
data_red.insert(0, 'redWins', first_column)

data_red['redFirstBlod'] = data_split['blueFirstBlood'].map({1:0,0:1})
first_column = data_red.pop('redFirstBlod')
data_red.insert(3, 'redFirstBlod', first_column)
```
```python
data_red.head(7)
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/41141a6ddb05253ab184f5dc722526756c9b7300/Screenshots/RedData_Pandas_dataFrame.png)

## Making plot charts so is easy to see the overall differences between teams 
```python
# First - the blue team
data_blue.drop(['blueGoldDiff', 'blueExperienceDiff', 'blueGoldDiff'], axis = 1).hist(color = 'b', figsize = (10,10))
plt.tight_layout()
plt.show()
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/41141a6ddb05253ab184f5dc722526756c9b7300/Screenshots/BlueData_Plot.png)
```python
# Then - the red team
data_red.hist(color = 'r', figsize = (10,10))
plt.tight_layout()
plt.show()
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/41141a6ddb05253ab184f5dc722526756c9b7300/Screenshots/RedData_Plot.png)
- It looks like there are no significant values that affect an especific team to win

```python
# Removing non-cuantificable data during match and reindexing columns for better readability
data = data.drop(['redKillDiff', 'blueExperienceDiff', 'redTotalJungleMinionsKilled',
                  'blueTotalJungleMinionsKilled', 'redWardsRemain','blueWardsRemain'], axis = 1)
                  
data = data.reindex(columns=['blueWins','blueFirstBlood', 'blueKillDiff', 'blueGoldDiff',
                             'blueAssists', 'blueDragons', 'blueHeralds','blueTowersDestroyed',
                             'redAssists', 'redDragons', 'redHeralds', 'redTowersDestroyed'])
data
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/38a366d4397083547c5867ee4646a9ef17c8d355/Screenshots/Reindex_data.png)
# Model
First, lets define the X and the Y values.
```python
X = data.drop("blueWins", axis = 1)
y = data["blueWins"]
```
Later, we set the Train-Test-Split function.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2907)
```
Then, we scale the data.
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
Now we can train the model
```python
from sklearn.linear_model import LogisticRegression

#Logistic Regression model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
```

Now that our model is fully trained, let's go and see how well does it made by seeing his classification report and other metrics to check his performance.

# Classification Report / Confusion Matrix / Score
```python
from sklearn.metrics import classification_report
from sklearn import metrics
pred = model_lr.predict(X_test)
```
## Classification Report
```python
# Class Report
print(classification_report(y_test,pred))
```
```python
OUTPUT:
              precision    recall  f1-score   support

           0       0.75      0.74      0.74       746
           1       0.74      0.75      0.74       736

    accuracy                           0.74      1482
   macro avg       0.74      0.74      0.74      1482
weighted avg       0.74      0.74      0.74      1482
```
## Confusion Matrix
```python
# Confusion Matrix
print("Confussion Matrix")
cm = metrics.confusion_matrix(y_test, pred)
print(cm)
```
Confussion Matrix

[[1119  378]

[ 389 1078]]

## Sklearn Metrics
```python
# Mean Absolute Error
print('MAE:', metrics.mean_absolute_error(y_test, pred))
# Mean Squared Error
print('MSE:', metrics.mean_squared_error(y_test, pred))
# Root Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
```
MAE: 0.25877192982456143

MSE: 0.25877192982456143

RMSE: 0.5086963041192274


## Score
```python
print("Score: {score}".format(score = model_lr.score(X_test, y_test)))
```
```python
OUTPUT:
  Score: 0.7422402159244265
```
# Prediction 
Lets try to make a random prediction
```python
import random

random.seed(2907)
random_ind = random.randint(0,len(data))

new_match = data.drop('blueWins',axis=1).iloc[random_ind]
new_match
```
```python
OUTPUT:
      blueFirstBlood            0
      blueKillDiff             -6
      blueGoldDiff          -4858
      blueAssists               3
      blueDragons               0
      blueHeralds               1
      blueTowersDestroyed       0
      redAssists               15
      redDragons                1
      redHeralds                0
      redTowersDestroyed        1
  Name: 9521, dtype: int64
```
```python
predictions = model_lr.predict(new_match.values.reshape(1,11))

threshold = 0.1
predictions = np.where(predictions > threshold, 1,0)

if predictions == 1:
    print('*Blue Team is going to win*')
else:
    print('*Red Team is going to win*')
```
```python
OUTPUT: 
  *Red Team is going to win*
```
# CONCLUSION
I am a League of Legends player for several years by now, and i consider that i have a great understanding of the data treated in here, so this project was very suitable for me, since i was looking for practice my knowledgement in data-cience and what better if not applying it to something that i really like and enjoy playing.

As my first Data-Cience project, i think this ended wrappin' up very nicely; it was very fun and appealing project to do, also, as a League player, i really found it really interesting by searching and viewing all the stats making sense when you compare them to each other for viewing how it affects if a team is going to win or not by the end of the match by simply watching the first ten minutes statistics.

Anyways, thanks for your time.
