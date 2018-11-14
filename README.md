# logistic-regression-python
#### Read in the data
```
import pandas as pd
myDF = pd.read_csv('wirelessdata.csv')
```
#### Show the data
```
myDF.head()
```
#### Check the number of rows
```
len(myDF)
```
#### If needed, get rid of rows with null / missing values - not necessary
```
myDF = myDF[pd.notnull(myDF['VU'])]
len(myDF)
```
#### Drop the unrequired variables
```
myDF = myDF.drop(['ValueUtility','ValueHedonistic', 'FeelDeviceDependence','ATUseSophistication','FeelTimeManagingDevice','UseSelfEfficacy'],axis=1)
myDF.head()
```
#### Import the packages
```
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
import numpy as np
```
#### Create matrices
```
y, X = dmatrices('VH ~ TIME + OWNED + SEX + AGE + \
                       Playfulness + NonPlayfulness', 
                       myDF,return_type= 'dataframe')

y= np.ravel(y) # if you don't do this you will get a warning in the next cell
```
#### sklearn output
```
model = LogisticRegression()
mdl = model.fit(X,y)
model.score(X,y)                # The "score" is stated in terms of "accurarcy"
```
0.66597938144329893

```
import statsmodels.discrete.discrete_model as sm
logit = sm.Logit(y,X)
logit.fit().params
```
Optimization terminated successfully.
         Current function value: 0.619339
         Iterations 5
Intercept        -3.345278
TIME             -0.001130
OWNED            -0.002455
SEX               0.795035
AGE               0.062777
Playfulness       0.145613
NonPlayfulness    0.040784
dtype: float64

#### Note that sex = 1,2--- 1 = female and 2 = male Increases in playful use and noPlayful use - both result in a corresponding increase in the likelihood of finding fun-related value in the smart phone. Time used and time owned result in a decrease in the likelihood of finding fun-related value in the smart phone.

#### What prcentage see fun-related value in their smart phones
```
y.mean()
```
0.41855670103092785
#### It seems approx 42% of smart phone users see fun-related value in their phones. This implies that 58% do not see fun-related value in their cell phones. In other words we could have obtained 52% accuracy by always predicting 0 (or 'no' for users seeing fun-related value in their smart phones.)

#### Split the train data and test data
```
import numpy as np
np.random.seed(0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train))
print(len(X_test))
```
339
146
#### C = Inverse of regularization strength; smaller values specify stronger regularization class_weight: wights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one dual : Dual or primal formulation fit_intercept : Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function. intercept_scaling: In case of imbalanced data (e.g. where the positive examples in the data are very rare) one can take a sample where the proportion of positives is higher. Training a logistic regression on this sample results in higher final predictions. The intercept scaling allows to convert the probabilities so that these reflect the initial data before sampling. max_iter: Maximum number of iterations taken for the solvers to converge. multi_class : Multiclass option can be either ‘ovr’ or ‘multinomial’. If the option chosen is ‘ovr’, then a binary problem is fit for each label. n_jobs : Number of CPU cores used when parallelizing over classes if multi_class=’ovr’. penalty : Used to specify the norm used in the penalization. L1-norm loss function is also known as least absolute deviations (LAD), least absolute errors (LAE). L2-norm loss function is also known as least squares error (LSE). random_state : The seed of the pseudo random number generator to use when shuffling the data solver : choice of solver from ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ tol : Tolerance for stopping criteria. verbose : it is generally an option for producing detailed logging information warm_start : When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
```
model2 = LogisticRegression()
model2.fit(X_train, y_train)
```
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

#### Usually the threshold is .5. The best threshold (or cutoff) point to be used in glm models is the point which maximises the specificity and the sensitivity.
```
predicted = model2.predict(X_test)
print (predicted)
```
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  1.  0.  1.  0.
  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
  0.  1.  1.  0.  0. 0.  1.  0.  0.  0.  0.  0.  0.  0.  1.  0.  1.  0.  0.  0.  0.  0.  0.
  0.  1.]
```
  myList = []
for i in range(len(predicted)):
    myList.append((int(y_test[i]),int(predicted[i])))
print(myList)
```
[(1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 1), (1, 0), (0, 1), (0, 0), (1, 1), (0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 1), (0, 1), (0, 0), (1, 0), (0, 1), (0, 1), (1, 1), (0, 0), (1, 1), (0, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 1), (0, 0), (1, 1), (1, 0), (0, 1), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 1), (1, 0), (1, 1), (1, 1), (1, 1), (1, 1), (0, 1), (0, 0), (0, 0), (0, 1), (0, 0), (1, 1), (0, 1), (0, 0), (0, 0), (0, 1), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0), (0, 1), (0, 1), (0, 1), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 1), (1, 0), (1, 0), (1, 0), (0, 0), (0, 1), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 0), (1, 0), (0, 0), (0, 0), (1, 1), (1, 1), (0, 0), (0, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 0), (0, 0), (1, 1), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0), (0, 1), (1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 1)]

#### generate class probabilities
```
probs = model2.predict_proba(X_test)
print (len(probs))
```
146

#### generate evaluation metrics
```
from sklearn import metrics
print (metrics.accuracy_score(y_test, predicted))
print (metrics.roc_auc_score(y_test, probs[:, 1]))
```
0.554794520548
0.588446969697
```
print (metrics.confusion_matrix(y_test, predicted))
print("-----------------")
print (metrics.classification_report(y_test, predicted))
```
[[58 22]
 [43 23]]
-----------------
             precision    recall  f1-score   support

        0.0       0.57      0.72      0.64        80
        1.0       0.51      0.35      0.41        66

avg / total       0.55      0.55      0.54       146


#### evaluate the model using 10-fold cross-validation
```
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())
```
[ 0.46        0.66        0.65306122  0.6875      0.5625      0.72916667
  0.625       0.64583333  0.64583333  0.66666667]
0.633556122449



# linear-regression-python
#### import the pandas,numpy,matplotlib.pyplot packages and load the data, read the data to check the variables.
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Read data
df = pd.read_csv('baseball.csv')
df.head()
```
##### PLOT the benchmark wins
```
mydf = df.loc[(df.Year >= 1996) & (df.Year <= 2001), ['Team', 'W', 'Playoffs']]
```
#### How many records in mydf

#### Get unique team names
```
team_list = list(mydf.Team.unique())
```
#### See what team_list looks like

#### Create team code column
```
import math
teamCode = []
```
#### For each row in the column,
```
for row in mydf['Team']:
    code = team_list.index(row)
    if code >= 0:
        teamCode.append(code)
    else:
        # Append a nan
        teamCode.append(math.nan)
``` 
#### Create a column in the data frame from the list
```
mydf['teamCode'] = teamCode
```
#### Create a new column 'Wins given gone to playoffs'
```
mydf['WPlayoffs'] = np.where(mydf['Playoffs']==1, mydf['W'], float('nan'))
```

#### Clear plot
```
plt.clf()
```
#### https://github.com/pydata/pandas/issues/9909
#### 2 y axes
#### secondary axis does not work on scatter plot
```
ax=mydf.plot(kind='scatter',x='teamCode',y='W',label='Wins',color='blue')
mydf.plot(kind='scatter',x='teamCode',y='WPlayoffs',label='Wins+Playoffs',color='red', \
    ax=ax,secondary_y=True)
plt.axhline(y=95, color = 'orange', linewidth=2)
plt.axhline(y=85, color = 'pink', linewidth=2)
plt.axhline(y=100, color = 'red', linewidth=2)
plt.show()
```
['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'FLA', 'HOU', 'KCR', 'LAD', 'MIL', 'MIN', 'MON', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBD', 'TEX', 'TOR', 'CAL']
    Team    W  Playoffs  teamCode  WPlayoffs
330  ANA   75         0         0        NaN
331  ARI   92         1         1       92.0
332  ATL   88         1         2       88.0
333  BAL   63         0         3        NaN
334  BOS   82         0         4        NaN
335  CHC   88         0         5        NaN
336  CHW   83         0         6        NaN
337  CIN   66         0         7        NaN
338  CLE   91         1         8       91.0
339  COL   73         0         9        NaN
340  DET   66         0        10        NaN
341  FLA   76         0        11        NaN
342  HOU   93         1        12       93.0
343  KCR   65         0        13        NaN
344  LAD   86         0        14        NaN
345  MIL   68         0        15        NaN
346  MIN   85         0        16        NaN
347  MON   68         0        17        NaN
348  NYM   82         0        18        NaN
349  NYY   95         1        19       95.0
350  OAK  102         1        20      102.0
351  PHI   86         0        21        NaN
352  PIT   62         0        22        NaN
353  SDP   79         0        23        NaN
354  SEA  116         1        24      116.0
355  SFG   90         0        25        NaN
356  STL   93         1        26       93.0
357  TBD   62         0        27        NaN
358  TEX   73         0        28        NaN
359  TOR   80         0        29        NaN
..   ...  ...       ...       ...        ...
476  TEX   77         0        28        NaN
477  TOR   76         0        29        NaN
478  ATL   96         1         2       96.0
479  BAL   88         1         3       88.0
480  BOS   85         0         4        NaN
481  CAL   70         0        30        NaN
482  CHC   76         0         5        NaN
483  CHW   85         0         6        NaN
484  CIN   81         0         7        NaN
485  CLE   99         1         8       99.0
486  COL   83         0         9        NaN
487  DET   53         0        10        NaN
488  FLA   80         0        11        NaN
489  HOU   82         0        12        NaN
490  KCR   75         0        13        NaN
491  LAD   90         1        14       90.0
492  MIL   80         0        15        NaN
493  MIN   78         0        16        NaN
494  MON   88         0        17        NaN
495  NYM   71         0        18        NaN
496  NYY   92         1        19       92.0
497  OAK   78         0        20        NaN
498  PHI   67         0        21        NaN
499  PIT   73         0        22        NaN
500  SDP   91         1        23       91.0
501  SEA   85         0        24        NaN
502  SFG   68         0        25        NaN
503  STL   88         1        26       88.0
504  TEX   90         1        28       90.0
505  TOR   74         0        29        NaN

[176 rows x 5 columns]
<matplotlib.figure.Figure at 0x28b43082160>

<img src="teamcode-linear-regression.png" class="img-responsive img-circle" alt="teacode">

```
df1 = df.query('Year < 2002')
newCol = df1.RS - df1.RA           # This is a data series
ds=newCol.to_frame()               # Convert this series to a data frame
ds.columns=['RD']                  # Name the column of this data frame
df1 = pd.concat([df1,ds],axis=1)   # pandas.pydata.org/pandas-docs/stable/merging.html
#df1.head()
df1[['RS','RA','RD']].head()
```
#### BEGIN of actual analysis

#### Method 2 - results in warnings
```
df1 = df.query('Year < 2002')
newCol = df1.RS - df1.RA
df1.loc[:,'RD'] = newCol
```
#### Method 3 - results in warnings
```
df1 = df.query('Year < 2002')
df1['RD'] = df1['RS']-df1['RA']
print(df1['RD'])
```
```
plt.clf()
plt.scatter(df1['RD'],df1['W'])
plt.xlabel('Run Difference')
plt.ylabel('Wins')
plt.title('Not so surprising')
plt.axis([-400, 400, 0, 120])
plt.text(0, 60, 'that it is a +ve relation', color='green', fontsize=15)
plt.grid(True)
plt.show()
```

<img src="difference-linear-regression.png" class="img-responsive img-circle" alt="difference">

#### Model 1
```
from sklearn import linear_model
from pandas import DataFrame

myData = DataFrame(data = df1, columns = ['W', 'RD'])
model = linear_model.LinearRegression(fit_intercept = True)
RD = myData.RD.values.reshape(len(myData), 1)
W = myData.W.values.reshape(len(myData), 1)
fit = model.fit(RD, W)
print ('Intercept: %.4f, Run Difference: %.4f' % (fit.intercept_, fit.coef_))

from sklearn.metrics import r2_score
pred = model.predict(RD)
r2 = r2_score(W,pred) 
print ('R-squared: %.4f' % (r2))
print()
print('R-squared: %.4f using alternate method' % fit.score(RD, W)) # Another way to get R2
```
Intercept: 80.8814, Run Difference: 0.1058
R-squared: 0.8808

R-squared: 0.8808 using alternate method

#### Model 2

```
x = DataFrame(data = df1, columns = ['OBP', 'SLG','BA'])
xarr = x.as_matrix()
X = np.array([np.concatenate((v,[1])) for v in xarr])
model = linear_model.LinearRegression(fit_intercept = True)
y = DataFrame(data = df1, columns = ['RS'])
yarr = y.as_matrix()
fit = model.fit(X,yarr)
#fit.intercept_
#fit.coef_

print("Intercept : ", fit.intercept_)
print("Slope : ", fit.coef_)

from sklearn.metrics import r2_score
pred = model.predict(X)
r2 = r2_score(yarr,pred) 
print ('R-squared: %.4f' % (r2))
```
Intercept :  [-788.45704708]
Slope :  [[ 2917.42140821  1637.92766577  -368.96606009     0.        ]]
R-squared: 0.9302

#### Remove BA due to reasons of multicollinearity
```
x = pd.DataFrame(data = df1, columns = ['OBP', 'SLG'])
xarr = x.as_matrix()
X = np.array([np.concatenate((v,[1])) for v in xarr])
model = linear_model.LinearRegression(fit_intercept = True)
y = DataFrame(data = df1, columns = ['RS'])
yarr = y.as_matrix()
fit = model.fit(X,yarr)
fit.intercept_
fit.coef_

from sklearn.metrics import r2_score
pred = model.predict(X)
r2 = r2_score(yarr, pred) 
print ('R-squared: %.4f' % (r2))
```
R-squared: 0.9296

#### Model 3
```
mydata = DataFrame(data = df1, columns = ['RA','OOBP', 'OSLG'])
len(mydata)
mydata1 = mydata.dropna()
len(mydata1)

x = DataFrame(data = mydata1, columns = ['OOBP', 'OSLG'])
xarr = x.as_matrix()
X = np.array([np.concatenate((v,[1])) for v in xarr])
model = linear_model.LinearRegression(fit_intercept = True)
y = DataFrame(data = mydata1, columns = ['RA'])
yarr = y.as_matrix()
fit = model.fit(X,yarr)
print("Intercept : ",fit.intercept_)
print("Slope : ", fit.coef_)

from sklearn.metrics import r2_score
pred = model.predict(X)
r2 = r2_score(yarr,pred) 
print ('R-squared: %.4f' % (r2))
```
Intercept :  [-837.37788861]
Slope :  [[ 2913.59948582  1514.28595842     0.        ]]
R-squared: 0.9073
