# Context
League of Legends is a MOBA (multiplayer online battle arena) where 2 teams (blue and red) face off. There are 3 lanes, a jungle, and 5 roles. The goal is to take down the enemy Nexus before the enemy destroys yours to win the game.

## Content
This dataset contains the first 10min. stats of approx. 10k ranked games (SOLO QUEUE) from a high ELO (DIAMOND I to MASTER). Players have roughly the same level.

Each game is unique. The gameId can help you to fetch more attributes from the Riot API.

There are 19 features per team (38 in total) collected after 10min in-game. This includes kills, deaths, gold, experience, level… It's up to you to do some feature engineering to get more insights.

The column blueWins is the target value (the value we are trying to predict). A value of 1 means the blue team has won. 0 otherwise.

So far I know, there is no missing value.

# Glossary
`Warding totem:` An item that a player can put on the map to reveal the nearby area. Very useful for map/objectives control.

`Minions:` NPC that belong to both teams. They give gold when killed by players.

`Jungle minions:` NPC that belong to NO TEAM. They give gold and buffs when killed by players.

`Elite monsters:` Monsters with high hp/damage that give a massive bonus (gold/XP/stats) when killed by a team.

`Dragons:` Elite monster which gives team bonus when killed. The 4th dragon killed by a team gives a massive stats bonus. The 5th dragon *(Elder Dragon)* offers a huge advantage to the team.

`Herald:` Elite monster which gives stats bonus when killed by the player. It helps to push a lane and destroys structures.

`Towers:` Structures you have to destroy to reach the enemy Nexus. They give gold.

`Level:` Champion level. Start at 1. Max is 18.


# Logistic_Regression
My Logisitic Regression aproach to predicting the result of League of Legends first 10 minutes matches 

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
data = pd.read_csv('/content/drive/MyDrive/GitHub/LoL_rankeds.csv')
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

## Separating the data by teams so i can look foword for further conclusions
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

Looking if there are matches in which a team is winning and then loses
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
#GameID isn't important for the algorithm so i decided to remove it
data = data.drop('gameId', axis=1)

# We drop all the variables that are symetric i.e blueKills == redDeaths
symetric_features = ["redFirstBlood", "blueDeaths", "redDeaths", "redGoldDiff", "redExperienceDiff"]

# We drop all the redundant variables. For example, we do not care about the total amount of gold earned, what matters is actually the difference compared to the enemy team
redundant_features = ['redTotalGold', 'blueTotalGold', 'redAvgLevel', 'blueAvgLevel', 'redTotalExperience', 'blueTotalExperience', 'redTotalMinionsKilled','blueTotalMinionsKilled', 'blueKills', 'redKills']

# We drop highly correlated features. For example, the redAvgLevel is logically highly corelated with the 
multicolinear_features = ["redCSPerMin", "blueCSPerMin", "blueEliteMonsters", "redEliteMonsters", "blueGoldPerMin", "redGoldPerMin"]

# Kill difference
data = data.assign(blueKillDiff = data["blueKills"] - data["redKills"])
data = data.assign(redKillDiff = data["redKills"] - data["blueKills"])
# Remaining wards
data = data.assign(blueWardsRemain = data["blueWardsPlaced"] - data["redWardsDestroyed"])
data = data.assign(redWardsRemain = data["redWardsPlaced"] - data["blueWardsDestroyed"])


data = data.drop(symetric_features, axis = 1)
data = data.drop(redundant_features, axis = 1)
data = data.drop(multicolinear_features, axis = 1)
```
Looking for high and low impact on determinating the Win condition 
```python
data.corr()["blueWins"].drop("blueWins").sort_values().plot(kind = "bar")
```
![alt text](https://github.com/JonathanCruze/League-of-Legends-Rankeds-Python-Analisis/blob/0f08521b5d28ed1c97ea0a3d18d658a48e6423ba/Screenshots/Corr_BlueWins_Plot.png)
```python
data.shape

  (9879, 22)
```
# Data Splitting
