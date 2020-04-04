### INTRODUCTION

- Data can be present in different ways.
- Types of data variables present in this data:
    - **Binary data** : A binary variable a variable that has only 2 values..ie 0/1
    - **Categorical data** : A categorical variable is a variable that can take some limited number of values.for example,day of the week.It can be one of 1,2,3,4,5,6,7 only.
    - **Ordinal data** : An ordinal variable is a categorical variable that has some order associated with it.for example,the ratings that are given to a movie by a user.
    - **Nominal data** : Nominal value is a variable that has no numerical importance,such as occupation,person name etc..
    - **Timeseries data** : Time series data has a temporal value attached to it, so this would be something like a date or a time stamp that you can look for trends in time.
![Label Encoding](https://techmintz.com/wp-content/uploads/2019/12/0*emSbyTsSeHaeFUKc-780x520.jpeg)
--- 
### Why do we need Label Encoding ?

- For example, a decision tree can be learned directly from categorical data with no data transform required (this depends on the specific implementation).

- Many machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric.

- In general, this is mostly a constraint of the efficient implementation of machine learning algorithms rather than hard limitations on the algorithms themselves.

- This means that categorical data must be converted to a numerical form. If the categorical variable is an output variable, you may also want to convert predictions by the model back into a categorical form in order to present them or use them in some application.
--- 
- In this notebook we will try some of the most commonly used encoding techniques.


### Downloading dataset from Kaggle using Kaggle API
- First, get kaggle.json file from your kaggle account and upload to gdrive.
- Second, Call Kaggle api to download the data 


```python
from google.colab import files
uploaded  = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

# Then move kaggle.json into the folder where the API expects to find it.
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```



<input type="file" id="files-9e6b332c-c7d8-4c24-99ff-1cb943bd4d32" name="files[]" multiple disabled />
<output id="result-9e6b332c-c7d8-4c24-99ff-1cb943bd4d32">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving kaggle.json to kaggle.json
    User uploaded file "kaggle.json" with length 69 bytes
    


```python
!kaggle competitions download -c cat-in-the-dat
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.6 / client 1.5.4)
    Downloading sample_submission.csv.zip to /content
      0% 0.00/436k [00:00<?, ?B/s]
    100% 436k/436k [00:00<00:00, 59.5MB/s]
    Downloading train.csv.zip to /content
     40% 5.00M/12.5M [00:00<00:00, 27.4MB/s]
    100% 12.5M/12.5M [00:00<00:00, 49.7MB/s]
    Downloading test.csv.zip to /content
     60% 5.00M/8.28M [00:00<00:00, 35.0MB/s]
    100% 8.28M/8.28M [00:00<00:00, 52.9MB/s]
    


```python
!unzip test.csv.zip
!unzip train.csv.zip
```

    Archive:  test.csv.zip
      inflating: test.csv                
    Archive:  train.csv.zip
      inflating: train.csv               
    

--- 

### Look at the data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import base
```


```python
df_train=pd.read_csv('/content/train.csv')
df_test=pd.read_csv('/content/test.csv')
```


```python
print('train data set has got {} rows and {} columns'.format(df_train.shape[0],df_train.shape[1]))
print('test data set has got {} rows and {} columns'.format(df_test.shape[0],df_test.shape[1]))
```

    train data set has got 300000 rows and 25 columns
    test data set has got 200000 rows and 24 columns
    


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bin_0</th>
      <th>bin_1</th>
      <th>bin_2</th>
      <th>bin_3</th>
      <th>bin_4</th>
      <th>nom_0</th>
      <th>nom_1</th>
      <th>nom_2</th>
      <th>nom_3</th>
      <th>nom_4</th>
      <th>nom_5</th>
      <th>nom_6</th>
      <th>nom_7</th>
      <th>nom_8</th>
      <th>nom_9</th>
      <th>ord_0</th>
      <th>ord_1</th>
      <th>ord_2</th>
      <th>ord_3</th>
      <th>ord_4</th>
      <th>ord_5</th>
      <th>day</th>
      <th>month</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>T</td>
      <td>Y</td>
      <td>Green</td>
      <td>Triangle</td>
      <td>Snake</td>
      <td>Finland</td>
      <td>Bassoon</td>
      <td>50f116bcf</td>
      <td>3ac1b8814</td>
      <td>68f6ad3e9</td>
      <td>c389000ab</td>
      <td>2f4cb3d51</td>
      <td>2</td>
      <td>Grandmaster</td>
      <td>Cold</td>
      <td>h</td>
      <td>D</td>
      <td>kr</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>T</td>
      <td>Y</td>
      <td>Green</td>
      <td>Trapezoid</td>
      <td>Hamster</td>
      <td>Russia</td>
      <td>Piano</td>
      <td>b3b4d25d0</td>
      <td>fbcb50fc1</td>
      <td>3b6dd5612</td>
      <td>4cd920251</td>
      <td>f83c56c21</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Hot</td>
      <td>a</td>
      <td>A</td>
      <td>bF</td>
      <td>7</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Blue</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Russia</td>
      <td>Theremin</td>
      <td>3263bdce5</td>
      <td>0922e3cb8</td>
      <td>a6a36f527</td>
      <td>de9c9f684</td>
      <td>ae6800dd0</td>
      <td>1</td>
      <td>Expert</td>
      <td>Lava Hot</td>
      <td>h</td>
      <td>R</td>
      <td>Jc</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Red</td>
      <td>Trapezoid</td>
      <td>Snake</td>
      <td>Canada</td>
      <td>Oboe</td>
      <td>f12246592</td>
      <td>50d7ad46a</td>
      <td>ec69236eb</td>
      <td>4ade6ab69</td>
      <td>8270f0d71</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Boiling Hot</td>
      <td>i</td>
      <td>D</td>
      <td>kW</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>N</td>
      <td>Red</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Canada</td>
      <td>Oboe</td>
      <td>5b0f5acd5</td>
      <td>1fe17a1fd</td>
      <td>04ddac2be</td>
      <td>cb43ab175</td>
      <td>b164b72a7</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Freezing</td>
      <td>a</td>
      <td>R</td>
      <td>qP</td>
      <td>7</td>
      <td>8</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Defining Training data and Target



```python
X=df_train.drop(['target'],axis=1)
y=df_train['target']
```


```python
x=y.value_counts()
plt.bar(x.index,x)
plt.gca().set_xticks([0,1])
plt.title('distribution of target variable')
plt.show()
```


![png](Label_Encoding_1_files/Label_Encoding_1_13_0.png)


--- 

### Method 1: Label encoding
- In this method we change every categorical data to a number
- That is each type will be replaced by a number
- For example we will substitute 1 for Grandmaster,2 for master ,3 for expert etc.. For implementing this we will first import Labelencoder from sklearn module


```python
%%time
from sklearn.preprocessing import LabelEncoder


train=pd.DataFrame()
label=LabelEncoder()
for c in  X.columns:
    if(X[c].dtype=='object'):
        train[c]=label.fit_transform(X[c])
    else:
        train[c]=X[c]
        
train.head(3)  
```

    CPU times: user 863 ms, sys: 27.8 ms, total: 891 ms
    Wall time: 892 ms
    



- Here you can see the label encoded output train data.We will check the shape of train data now and verify that there is no change in the number of columns.




```python
print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
```

    train data set has got 300000 rows and 24 columns
    

#### Logistic Regression:
- We will use logistic regression to predict the target label on label encoded data


```python
def logistic(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    y_pre=lr.predict(X_test)
    print('Accuracy : ',accuracy_score(y_test,y_pre))
```


```python
logistic(train,y)
```

    Accuracy :  0.69065
    

### Method 2 : On hot encoding
- This type of encoding converts each category as a one hot encoding (OHE) vector (or dummy variables). OHE is a representation method that takes each category value and turns it into a binary vector of size |i|(number of values in category i) where all columns are equal to zero besides the category column. 
- Here is a little example:
![One Hot Encoding](https://miro.medium.com/max/878/1*WXpoiS7HXRC-uwJPYsy1Dg.png)


```python
%%time
from sklearn.preprocessing import OneHotEncoder


one=OneHotEncoder()
one.fit(X)
train=one.transform(X)

print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
```

    train data set has got 300000 rows and 316461 columns
    CPU times: user 1.49 s, sys: 31.8 ms, total: 1.52 s
    Wall time: 1.52 s
    


```python
logistic(train,y)
```

    Accuracy :  0.75715
    

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

### Method 3 : Feature hashing (hashing trick)

- Feature hashing is a very cool technique to represent categories in a “one hot encoding style” as a sparse matrix but with a much lower dimensions. 
- In feature hashing we apply a hashing function to the category and then represent it by its indices. 
- For example, if we choose a dimension of 5 to represent “New York” we will calculate H(New York) mod 5 = 3 (for example) so New York representation will be (0,0,1,0,0).



```python
%%time
from sklearn.feature_extraction import FeatureHasher



X_train_hash=X.copy()
for c in X.columns:
    X_train_hash[c]=X[c].astype('str')      
hashing=FeatureHasher(input_type='string')
train=hashing.transform(X_train_hash.values)
```

    CPU times: user 4.08 s, sys: 63.6 ms, total: 4.14 s
    Wall time: 4.16 s
    


```python
print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
```

    train data set has got 300000 rows and 1048576 columns
    


```python
logistic(train,y)
```

    Accuracy :  0.7516833333333334
    

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

### Method 4 : Encoding categories with dataset statistic

- Now we will try to give our models a numeric representation for every category with a small number of columns but with an encoding that will put similar categories close to each other. 
- The easiest way to do it is replace every category with the number of times that we saw it in the dataset. 
- This way if New York and New Jersey are both big cities, they will probably both appear many times in our dataset and the model will know that they are similar.


```python
%%time

X_train_stat=X.copy()
for c in X_train_stat.columns:
    if(X_train_stat[c].dtype=='object'):
        X_train_stat[c]=X_train_stat[c].astype('category')
        counts=X_train_stat[c].value_counts()
        counts=counts.sort_index()
        counts=counts.fillna(0)
        counts += np.random.rand(len(counts))/1000
        X_train_stat[c].cat.categories=counts
```

    CPU times: user 562 ms, sys: 14.3 ms, total: 576 ms
    Wall time: 596 ms
    


```python
X_train_stat.head(3)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bin_0</th>
      <th>bin_1</th>
      <th>bin_2</th>
      <th>bin_3</th>
      <th>bin_4</th>
      <th>nom_0</th>
      <th>nom_1</th>
      <th>nom_2</th>
      <th>nom_3</th>
      <th>nom_4</th>
      <th>nom_5</th>
      <th>nom_6</th>
      <th>nom_7</th>
      <th>nom_8</th>
      <th>nom_9</th>
      <th>ord_0</th>
      <th>ord_1</th>
      <th>ord_2</th>
      <th>ord_3</th>
      <th>ord_4</th>
      <th>ord_5</th>
      <th>day</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>153535.000307</td>
      <td>191633.000349</td>
      <td>127341.000613</td>
      <td>29855.000407</td>
      <td>45979.000524</td>
      <td>36942.000669</td>
      <td>68448.000619</td>
      <td>2594.000460</td>
      <td>1148.000100</td>
      <td>241.000134</td>
      <td>271.000419</td>
      <td>19.000890</td>
      <td>2</td>
      <td>77428.000840</td>
      <td>33768.000056</td>
      <td>24740.000813</td>
      <td>3974.000743</td>
      <td>506.000402</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>153535.000307</td>
      <td>191633.000349</td>
      <td>127341.000613</td>
      <td>101181.000097</td>
      <td>29487.000694</td>
      <td>101123.000021</td>
      <td>84517.000418</td>
      <td>792.000803</td>
      <td>842.000982</td>
      <td>287.000052</td>
      <td>111.000134</td>
      <td>13.000259</td>
      <td>1</td>
      <td>77428.000840</td>
      <td>22227.000246</td>
      <td>35276.000920</td>
      <td>18258.000000</td>
      <td>2603.000699</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>146465.000435</td>
      <td>191633.000349</td>
      <td>96166.000929</td>
      <td>101181.000097</td>
      <td>101295.000356</td>
      <td>101123.000021</td>
      <td>54742.000541</td>
      <td>2524.000536</td>
      <td>1169.000602</td>
      <td>475.000152</td>
      <td>278.000688</td>
      <td>29.000682</td>
      <td>1</td>
      <td>25065.000477</td>
      <td>63908.000183</td>
      <td>24740.000813</td>
      <td>16927.000531</td>
      <td>2572.000440</td>
      <td>7</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('train data set has got {} rows and {} columns'.format(X_train_stat.shape[0],X_train_stat.shape[1]))
```

    train data set has got 300000 rows and 24 columns
    


```python
logistic(X_train_stat,y)
```

    Accuracy :  0.6946
    

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

### Encoding cyclic features

![cyclic features encoding](https://miro.medium.com/max/343/1*70cevmU8wNggGJEdLam1lw.png)

- Some of our features are cyclic in nature.ie day,month etc.

- A common method for encoding cyclical data is to transform the data into two dimensions using a sine and consine transformation.


```python
%%time

X_train_cyclic=X.copy()
columns=['day','month']
for col in columns:
    X_train_cyclic[col+'_sin']=np.sin((2*np.pi*X_train_cyclic[col])/max(X_train_cyclic[col]))
    X_train_cyclic[col+'_cos']=np.cos((2*np.pi*X_train_cyclic[col])/max(X_train_cyclic[col]))
X_train_cyclic=X_train_cyclic.drop(columns,axis=1)

X_train_cyclic[['day_sin','day_cos']].head(3)
```

    CPU times: user 256 ms, sys: 1.74 ms, total: 258 ms
    Wall time: 254 ms
    

- Now we will use OnHotEncoder to encode other variables,then feed the data to our model.


```python
one=OneHotEncoder()

one.fit(X_train_cyclic)
train=one.transform(X_train_cyclic)

print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
```

    train data set has got 300000 rows and 316478 columns
    


```python
logistic(train,y)
```

    Accuracy :  0.758
    

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    

### Method 5 : Target encoding

- Target-based encoding is numerization of categorical variables via target. 
- In this method, we replace the categorical variable with just one new numerical variable and replace each category of the categorical variable with its corresponding probability of the target (if categorical) or average of the target (if numerical). 
- The main drawbacks of this method are its dependency to the distribution of the target, and its lower predictability power compare to the binary encoding method.

- For example,

| Country | Target |
| --- | --- | 
| India  | 1 | 
| China | 0 | 
| India | 0 | 
| China | 1 | 
| India | 1 | 

- Encoding for India = [Number of true targets under the label India/ Total Number of targets under the label India] which is 2/3 = 0.66

| Country | Target |
| --- | --- | 
| India  | 0.66 | 
| China | 0.5 | 





```python
%%time

X_target=df_train.copy()
X_target['day']=X_target['day'].astype('object')
X_target['month']=X_target['month'].astype('object')
for col in X_target.columns:
    if (X_target[col].dtype=='object'):
        target= dict ( X_target.groupby(col)['target'].agg('sum')/X_target.groupby(col)['target'].agg('count'))
        X_target[col]=X_target[col].replace(target).values
        
X_target.head(4)
```

    CPU times: user 2min 24s, sys: 48.7 s, total: 3min 13s
    Wall time: 3min 13s
    


```python
logistic(X_target.drop('target',axis=1),y)
```

    Accuracy :  0.6946166666666667
    

### Summary - 

- Here you can see the summary of our model performance against each of the encoding techniques we have used. 
- It is clear that OnHotEncoder together with cyclic feature encoding yielded maximum accuracy.


| ENCODING | SCORE | WALLTIME |
| --- | --- | --- |
|Label Encoding	|0.692	|892 ms|
|OnHotEncoder	|0.759	|1.52 s|
|Feature Hashing|	0.751	|4.16 s|
|Dataset statistic encoding	|0.694	|596 ms|
|Cyclic + OnHotEncoding	|0.759|	254 ms|
|Target encoding|	0.694	|3min 13s|

