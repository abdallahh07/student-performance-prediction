import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error

# 1. Date
f= pd.read_csv("student_performance.csv")

# 2. reviewing the data
print(f.head(5))
print(f.dtypes)
print(f.info())
print(f.describe())

# 3. Analyze the realtion between the data 
def analyze(f):
  
  fig,ax = plt.subplots(3,2,figsize=(15,10))
  ax=ax.flatten()
  
  # 3.1 Attendance vs Previous Score
  sns.scatterplot(data=f,
                  x="attendance_rate",
                  y="previous_score",
                  ax=ax[0])
  
  # 3.2 study hours distributions
  sns.histplot(data=f["study_hours_per_week"],
               bins=30,
               kde=True,
               ax=ax[1])
  
  # 3.3 Gender Distribution
  sns.countplot(data=f, x="gender", ax=ax[2])
  
  # 3.4 Sleep Hours vs Exam Score
  sns.barplot(data=f,
                  x="internet_access",
                  y="final_score",
                  ax=ax[3])
  
  # 3.5 Age distribution
  sns.histplot(f["age"],
               bins=20,
               kde=True,
               ax=ax[4])
  
  # 3.6 final scores 
  sns.histplot(f["final_score"],
              bins=20,
              kde=True,
              ax=ax[5])
  ax[5].set_title("Final Score Distribution")
  
  plt.show()

analyze(f)
# 4. filling the null values
print(f.isna().sum())

impute=SimpleImputer(strategy="most_frequent")
f[["parent_education"]] = impute.fit_transform(f[["parent_education"]])

# 5. Data Preprocessing
one_hot=OneHotEncoder(handle_unknown="ignore")
categorical_feature=["gender","parent_education","internet_access",
                     "extracurricular"]
transformer=ColumnTransformer([("one_hot",
                                one_hot,
                                categorical_feature,)],
                              remainder="passthrough")

x=f.drop(["final_score","student_id","passed"],axis=1)
y=f["final_score"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# 6. Models
models={
  "Linear Regression":LinearRegression(),
  "Decision Tree" : DecisionTreeRegressor(random_state=42),
  "Random Forest" : RandomForestRegressor(random_state=42),
  "XGBoost":XGBRegressor(random_state=42)}

result=[]
for name , model in models.items():
  pipe=Pipeline([
    ("Preprocessing",transformer),
    ("Scaler",StandardScaler(with_mean=False)),
    ("Poly",PolynomialFeatures(degree=2,include_bias=False)),
    ("Model",model)])

# 7. Train & Evaluate 
  pipe.fit(x_train,y_train)
  pred = pipe.predict(x_test)

  print(f"\n{name}")
  print("R2 Score:",r2_score(y_test,pred))
  print("MSE:",mean_squared_error(y_test,pred))
  print("RMSE:",root_mean_squared_error(y_test,pred))
  
  result.append({
    "Model":name,
    "R2 Score":r2_score(y_test,pred)
  })
# 8. vizualize the models accuracy
model_res=pd.DataFrame(result)   
plt.figure(figsize=(10,6))
sns.barplot(data=model_res,
  x="Model",
  y="R2 Score")


plt.title("Model R2 Comparison")
plt.ylabel("R2 Score")
plt.show()