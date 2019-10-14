import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# We are reading our data
df = pd.read_csv("../recons_dataset/combined_dataset.csv")
df.head()

# %% [markdown]
# Data contains; <br>
#
# * age - age in years <br>
# * sex - (1 = male; 0 = female) <br>
# * cp - chest pain type <br>
# * trestbps - resting blood pressure (in mm Hg on admission to the hospital) <br>
# * chol - serum cholestoral in mg/dl <br>
# * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) <br>
# * restecg - resting electrocardiographic results <br>
# * thalach - maximum heart rate achieved <br>
# * exang - exercise induced angina (1 = yes; 0 = no) <br>
# * ca - number of major vessels (0-3) colored by flourosopy <br>
# * thal - 3 = normal; 6 = fixed defect; 7 = reversable defect <br>
# * target - have disease or not (1=yes, 0=no)


# %% [code]
df.target.value_counts()

# %% [code]
sns.countplot(x="target", data=df, palette="bwr")
plt.show()

# %% [code]
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))

# %% [code]
countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))

# %% [code]
df.groupby('target').mean()

# %% [code]
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()

# %% [code]
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

# %% [code]
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

# %% [code]
pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()

# %% [code]
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()

# %% [code]
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()

# %% [markdown]
# ### Creating Dummy Variables

# %% [markdown]
# ![](http://)Since 'cp', 'thal' and 'slope' are categorical variables we'll turn them into dummy variables.

# %% [code]
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

# %% [code]
frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()
