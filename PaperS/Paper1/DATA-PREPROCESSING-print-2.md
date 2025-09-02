# Data Preprocessing


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from feature_engine.creation import CyclicalFeatures
from sklearn.preprocessing import RobustScaler
```


```python
data = pd.read_csv('cleanest_data.csv')
```

## Попередній огляд даних


```python
print(data)
```

           is_successful                 create_date  order_amount  \
    0                  1  2017-07-29 07:48:26.812523       5235.66   
    1                  1  2017-07-29 07:54:09.954757        876.96   
    2                  1  2017-07-29 08:04:13.162858       3012.77   
    3                  1  2017-07-29 08:11:38.086709        621.34   
    4                  1  2017-07-29 08:15:05.548616        813.12   
    ...              ...                         ...           ...   
    86789              1  2024-12-11 11:09:57.124395          0.53   
    86790              0  2024-12-16 08:38:35.387458        697.37   
    86791              1  2024-12-19 13:59:08.130686        129.96   
    86792              1  2025-01-02 08:33:33.424152        129.96   
    86793              1  2025-01-02 08:36:13.298285        190.44   
    
           order_messages  order_changes  partner_success_rate  \
    0                  25             22              0.000000   
    1                  10              5              0.000000   
    2                   7              4              0.000000   
    3                  10              6              0.000000   
    4                   6              3              0.000000   
    ...               ...            ...                   ...   
    86789               5              2             11.111111   
    86790               2              0            100.000000   
    86791               2              1              0.000000   
    86792               2              1            100.000000   
    86793               4              2            100.000000   
    
           partner_total_orders  partner_order_age_days  partner_avg_amount  \
    0                         0                       0            0.000000   
    1                         0                       0            0.000000   
    2                         0                       0            0.000000   
    3                         0                       0            0.000000   
    4                         0                       0            0.000000   
    ...                     ...                     ...                 ...   
    86789                     9                      57        71388.001111   
    86790                    24                     752          530.264583   
    86791                     0                       0            0.000000   
    86792                     1                      13          129.960000   
    86793                     7                     933          286.988571   
    
           partner_success_avg_amount  ...  partner_success_avg_changes  \
    0                        0.000000  ...                     0.000000   
    1                        0.000000  ...                     0.000000   
    2                        0.000000  ...                     0.000000   
    3                        0.000000  ...                     0.000000   
    4                        0.000000  ...                     0.000000   
    ...                           ...  ...                          ...   
    86789                    0.000000  ...                     4.000000   
    86790                  530.264583  ...                     2.458333   
    86791                    0.000000  ...                     0.000000   
    86792                  129.960000  ...                     1.000000   
    86793                  286.988571  ...                     3.857143   
    
           partner_fail_avg_changes  day_of_week     month  quarter  hour_of_day  \
    0                         0.000     Saturday      July        3            7   
    1                         0.000     Saturday      July        3            7   
    2                         0.000     Saturday      July        3            8   
    3                         0.000     Saturday      July        3            8   
    4                         0.000     Saturday      July        3            8   
    ...                         ...          ...       ...      ...          ...   
    86789                     3.375    Wednesday  December        4           11   
    86790                     0.000       Monday  December        4            8   
    86791                     0.000     Thursday  December        4           13   
    86792                     0.000     Thursday   January        1            8   
    86793                     0.000     Thursday   January        1            8   
    
           order_lines_count discount_total salesperson  source  
    0                      6            0.0   user-1-76   False  
    1                      3            0.0   user-1-76   False  
    2                      4            0.0    user-1-9   False  
    3                      4            0.0    user-1-2   False  
    4                      3            0.0    user-1-9   False  
    ...                  ...            ...         ...     ...  
    86789                  1            0.0   user-1-49   False  
    86790                  3            0.0  user-1-113   False  
    86791                  1            0.0    user-1-2   False  
    86792                  1            0.0    user-1-2   False  
    86793                  2            0.0   user-1-10   False  
    
    [86794 rows x 25 columns]
    


```python
print(data.head(10))
```

       is_successful                 create_date  order_amount  order_messages  \
    0              1  2017-07-29 07:48:26.812523       5235.66              25   
    1              1  2017-07-29 07:54:09.954757        876.96              10   
    2              1  2017-07-29 08:04:13.162858       3012.77               7   
    3              1  2017-07-29 08:11:38.086709        621.34              10   
    4              1  2017-07-29 08:15:05.548616        813.12               6   
    5              1  2017-07-29 08:19:38.625071       1029.62              22   
    6              1  2017-07-29 08:29:50.487564       2832.00              10   
    7              1  2017-07-29 08:40:21.407789        340.78              10   
    8              1  2017-07-29 08:44:45.461359       1546.80              11   
    9              1  2017-07-29 08:53:03.880701       2080.80               8   
    
       order_changes  partner_success_rate  partner_total_orders  \
    0             22                   0.0                     0   
    1              5                   0.0                     0   
    2              4                   0.0                     0   
    3              6                   0.0                     0   
    4              3                   0.0                     0   
    5             19                   0.0                     0   
    6              7                   0.0                     0   
    7              6                   0.0                     0   
    8              8                   0.0                     0   
    9              4                   0.0                     0   
    
       partner_order_age_days  partner_avg_amount  partner_success_avg_amount  \
    0                       0                 0.0                         0.0   
    1                       0                 0.0                         0.0   
    2                       0                 0.0                         0.0   
    3                       0                 0.0                         0.0   
    4                       0                 0.0                         0.0   
    5                       0                 0.0                         0.0   
    6                       0                 0.0                         0.0   
    7                       0                 0.0                         0.0   
    8                       0                 0.0                         0.0   
    9                       0                 0.0                         0.0   
    
       ...  partner_success_avg_changes  partner_fail_avg_changes  day_of_week  \
    0  ...                          0.0                       0.0     Saturday   
    1  ...                          0.0                       0.0     Saturday   
    2  ...                          0.0                       0.0     Saturday   
    3  ...                          0.0                       0.0     Saturday   
    4  ...                          0.0                       0.0     Saturday   
    5  ...                          0.0                       0.0     Saturday   
    6  ...                          0.0                       0.0     Saturday   
    7  ...                          0.0                       0.0     Saturday   
    8  ...                          0.0                       0.0     Saturday   
    9  ...                          0.0                       0.0     Saturday   
    
       month  quarter  hour_of_day  order_lines_count discount_total salesperson  \
    0   July        3            7                  6            0.0   user-1-76   
    1   July        3            7                  3            0.0   user-1-76   
    2   July        3            8                  4            0.0    user-1-9   
    3   July        3            8                  4            0.0    user-1-2   
    4   July        3            8                  3            0.0    user-1-9   
    5   July        3            8                  2            0.0    user-1-9   
    6   July        3            8                  2            0.0   user-1-76   
    7   July        3            8                  3            0.0    user-1-9   
    8   July        3            8                  2            0.0    user-1-9   
    9   July        3            8                  3            0.0    user-1-9   
    
       source  
    0   False  
    1   False  
    2   False  
    3   False  
    4   False  
    5   False  
    6   False  
    7   False  
    8   False  
    9   False  
    
    [10 rows x 25 columns]
    


```python
print(data.tail(10))
```

           is_successful                 create_date  order_amount  \
    86784              1  2024-11-26 09:23:06.451541         81.67   
    86785              0  2024-11-26 09:34:52.576840          0.00   
    86786              0  2024-12-04 15:14:51.956392          0.00   
    86787              1  2024-12-09 10:22:13.166600        203.96   
    86788              1  2024-12-10 11:07:58.049169        108.30   
    86789              1  2024-12-11 11:09:57.124395          0.53   
    86790              0  2024-12-16 08:38:35.387458        697.37   
    86791              1  2024-12-19 13:59:08.130686        129.96   
    86792              1  2025-01-02 08:33:33.424152        129.96   
    86793              1  2025-01-02 08:36:13.298285        190.44   
    
           order_messages  order_changes  partner_success_rate  \
    86784               2              1             33.333333   
    86785               1              0              0.000000   
    86786               1              0             58.196721   
    86787               2              1             81.651376   
    86788               5              4              0.000000   
    86789               5              2             11.111111   
    86790               2              0            100.000000   
    86791               2              1              0.000000   
    86792               2              1            100.000000   
    86793               4              2            100.000000   
    
           partner_total_orders  partner_order_age_days  partner_avg_amount  \
    86784                     3                     453           27.410000   
    86785                     0                       0            0.000000   
    86786                   122                    2158         4455.098197   
    86787                   109                    2685         6732.654220   
    86788                     0                       0            0.000000   
    86789                     9                      57        71388.001111   
    86790                    24                     752          530.264583   
    86791                     0                       0            0.000000   
    86792                     1                      13          129.960000   
    86793                     7                     933          286.988571   
    
           partner_success_avg_amount  ...  partner_success_avg_changes  \
    86784                   81.670000  ...                     1.000000   
    86785                    0.000000  ...                     0.000000   
    86786                 3069.750704  ...                     4.676056   
    86787                 5564.707528  ...                     7.101124   
    86788                    0.000000  ...                     0.000000   
    86789                    0.000000  ...                     4.000000   
    86790                  530.264583  ...                     2.458333   
    86791                    0.000000  ...                     0.000000   
    86792                  129.960000  ...                     1.000000   
    86793                  286.988571  ...                     3.857143   
    
           partner_fail_avg_changes  day_of_week     month  quarter  hour_of_day  \
    86784                  1.500000      Tuesday  November        4            9   
    86785                  0.000000      Tuesday  November        4            9   
    86786                  1.901961    Wednesday  December        4           15   
    86787                  5.450000       Monday  December        4           10   
    86788                  0.000000      Tuesday  December        4           11   
    86789                  3.375000    Wednesday  December        4           11   
    86790                  0.000000       Monday  December        4            8   
    86791                  0.000000     Thursday  December        4           13   
    86792                  0.000000     Thursday   January        1            8   
    86793                  0.000000     Thursday   January        1            8   
    
           order_lines_count discount_total salesperson  source  
    86784                  1            0.0    user-1-9   False  
    86785                  0            0.0   user-1-39   False  
    86786                  0            0.0    user-1-9   False  
    86787                  2            0.0   user-1-10   False  
    86788                  1            0.0   user-1-39   False  
    86789                  1            0.0   user-1-49   False  
    86790                  3            0.0  user-1-113   False  
    86791                  1            0.0    user-1-2   False  
    86792                  1            0.0    user-1-2   False  
    86793                  2            0.0   user-1-10   False  
    
    [10 rows x 25 columns]
    


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 86794 entries, 0 to 86793
    Data columns (total 25 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   is_successful                 86794 non-null  int64  
     1   create_date                   86794 non-null  object 
     2   order_amount                  86794 non-null  float64
     3   order_messages                86794 non-null  int64  
     4   order_changes                 86794 non-null  int64  
     5   partner_success_rate          86794 non-null  float64
     6   partner_total_orders          86794 non-null  int64  
     7   partner_order_age_days        86794 non-null  int64  
     8   partner_avg_amount            86794 non-null  float64
     9   partner_success_avg_amount    86794 non-null  float64
     10  partner_fail_avg_amount       86794 non-null  float64
     11  partner_total_messages        86794 non-null  int64  
     12  partner_success_avg_messages  86794 non-null  float64
     13  partner_fail_avg_messages     86794 non-null  float64
     14  partner_avg_changes           86794 non-null  float64
     15  partner_success_avg_changes   86794 non-null  float64
     16  partner_fail_avg_changes      86794 non-null  float64
     17  day_of_week                   86794 non-null  object 
     18  month                         86794 non-null  object 
     19  quarter                       86794 non-null  int64  
     20  hour_of_day                   86794 non-null  int64  
     21  order_lines_count             86794 non-null  int64  
     22  discount_total                86794 non-null  float64
     23  salesperson                   86794 non-null  object 
     24  source                        86794 non-null  object 
    dtypes: float64(11), int64(9), object(5)
    memory usage: 16.6+ MB
    


```python
print(data.describe())
```

           is_successful  order_amount  order_messages  order_changes  \
    count   86794.000000  8.679400e+04    86794.000000   86794.000000   
    mean        0.630954  3.824407e+03        9.408542       3.904521   
    std         0.482549  3.141098e+04        8.222635       3.836411   
    min         0.000000 -6.242400e+03        1.000000       0.000000   
    25%         0.000000  4.012500e+00        5.000000       2.000000   
    50%         1.000000  6.159850e+02        7.000000       3.000000   
    75%         1.000000  1.912025e+03       11.000000       4.000000   
    max         1.000000  4.140000e+06      282.000000     274.000000   
    
           partner_success_rate  partner_total_orders  partner_order_age_days  \
    count          86794.000000          86794.000000            86794.000000   
    mean              59.600411             83.696258              899.232136   
    std               31.305902            151.012261              754.145810   
    min                0.000000              0.000000                0.000000   
    25%               44.186047              5.000000              219.000000   
    50%               66.666667             26.000000              764.000000   
    75%               83.333333             90.000000             1430.000000   
    max              100.000000           1307.000000             2685.000000   
    
           partner_avg_amount  partner_success_avg_amount  \
    count        8.679400e+04                86794.000000   
    mean         2.914209e+03                 1597.019479   
    std          1.393899e+04                 3562.194344   
    min          0.000000e+00                    0.000000   
    25%          4.738430e+02                  430.711667   
    50%          1.127518e+03                  933.960000   
    75%          2.545909e+03                 1789.247500   
    max          2.700000e+06               119310.000000   
    
           partner_fail_avg_amount  partner_total_messages  \
    count             8.679400e+04            86794.000000   
    mean              4.112985e+03              758.382688   
    std               1.946169e+04             1372.013756   
    min               0.000000e+00                0.000000   
    25%               0.000000e+00               47.000000   
    50%               8.216000e+02              240.000000   
    75%               3.337202e+03              803.000000   
    max               2.700000e+06            11481.000000   
    
           partner_success_avg_messages  partner_fail_avg_messages  \
    count                  86794.000000               86794.000000   
    mean                       9.556764                   5.705881   
    std                        5.916261                   4.532492   
    min                        0.000000                   0.000000   
    25%                        7.454545                   4.555556   
    50%                        9.474948                   5.666667   
    75%                       12.041667                   6.937500   
    max                      123.500000                 143.000000   
    
           partner_avg_changes  partner_success_avg_changes  \
    count         86794.000000                 86794.000000   
    mean              4.060877                     4.409310   
    std               2.372341                     2.945225   
    min               0.000000                     0.000000   
    25%               3.045045                     3.156250   
    50%               3.969032                     4.333333   
    75%               4.961039                     5.625000   
    max             137.500000                    57.000000   
    
           partner_fail_avg_changes       quarter   hour_of_day  \
    count              86794.000000  86794.000000  86794.000000   
    mean                   3.058853      2.472821     10.937000   
    std                    2.966175      1.108799      2.888376   
    min                    0.000000      1.000000      0.000000   
    25%                    2.000000      1.000000      9.000000   
    50%                    2.958333      2.000000     11.000000   
    75%                    3.666667      3.000000     13.000000   
    max                  137.500000      4.000000     23.000000   
    
           order_lines_count  discount_total  
    count       86794.000000    86794.000000  
    mean            3.369645        0.039169  
    std             5.282779        2.226804  
    min             0.000000        0.000000  
    25%             2.000000        0.000000  
    50%             3.000000        0.000000  
    75%             4.000000        0.000000  
    max          1304.000000      390.000000  
    


```python
print(data.nunique())
```

    is_successful                       2
    create_date                     86791
    order_amount                    49339
    order_messages                    132
    order_changes                      70
    partner_success_rate            12791
    partner_total_orders             1308
    partner_order_age_days           2667
    partner_avg_amount              75712
    partner_success_avg_amount      49415
    partner_fail_avg_amount         24093
    partner_total_messages           6297
    partner_success_avg_messages    18883
    partner_fail_avg_messages        6294
    partner_avg_changes             23981
    partner_success_avg_changes     15727
    partner_fail_avg_changes         5687
    day_of_week                         7
    month                              12
    quarter                             4
    hour_of_day                        24
    order_lines_count                  59
    discount_total                     36
    salesperson                        31
    source                             66
    dtype: int64
    


```python
print(data.isnull().sum())
```

    is_successful                   0
    create_date                     0
    order_amount                    0
    order_messages                  0
    order_changes                   0
    partner_success_rate            0
    partner_total_orders            0
    partner_order_age_days          0
    partner_avg_amount              0
    partner_success_avg_amount      0
    partner_fail_avg_amount         0
    partner_total_messages          0
    partner_success_avg_messages    0
    partner_fail_avg_messages       0
    partner_avg_changes             0
    partner_success_avg_changes     0
    partner_fail_avg_changes        0
    day_of_week                     0
    month                           0
    quarter                         0
    hour_of_day                     0
    order_lines_count               0
    discount_total                  0
    salesperson                     0
    source                          0
    dtype: int64
    


```python
print(data['is_successful'].value_counts(normalize=True))
```

    is_successful
    1    0.630954
    0    0.369046
    Name: proportion, dtype: float64
    


```python
print(pd.crosstab(data['day_of_week'], data['is_successful'], normalize='index'))
```

    is_successful         0         1
    day_of_week                      
    Friday         0.386884  0.613116
    Monday         0.353468  0.646532
    Saturday       0.294964  0.705036
    Sunday         0.278689  0.721311
    Thursday       0.363158  0.636842
    Tuesday        0.375406  0.624594
    Wednesday      0.368458  0.631542
    


```python
print(pd.crosstab(data['month'], data['is_successful'], normalize='index'))
```

    is_successful         0         1
    month                            
    April          0.401789  0.598211
    August         0.344703  0.655297
    December       0.403068  0.596932
    February       0.402014  0.597986
    January        0.360267  0.639733
    July           0.362590  0.637410
    June           0.365844  0.634156
    March          0.377981  0.622019
    May            0.365255  0.634745
    November       0.355967  0.644033
    October        0.349327  0.650673
    September      0.356758  0.643242
    


```python
print(pd.crosstab(data['salesperson'], data['is_successful'], normalize='index'))
```

    is_successful         0         1
    salesperson                      
    user-1-10      0.294904  0.705096
    user-1-100     0.400000  0.600000
    user-1-1066    0.500000  0.500000
    user-1-11      0.047619  0.952381
    user-1-113     0.299200  0.700800
    user-1-1366    0.214286  0.785714
    user-1-14      0.150794  0.849206
    user-1-1414    0.312169  0.687831
    user-1-142     0.290053  0.709947
    user-1-1451    0.000000  1.000000
    user-1-1465    1.000000  0.000000
    user-1-1582    0.000000  1.000000
    user-1-19      0.291667  0.708333
    user-1-2       0.459156  0.540844
    user-1-39      0.500000  0.500000
    user-1-4       0.111111  0.888889
    user-1-47      0.233333  0.766667
    user-1-49      0.384541  0.615459
    user-1-54      0.400000  0.600000
    user-1-56      0.500000  0.500000
    user-1-63      0.000000  1.000000
    user-1-67      0.090278  0.909722
    user-1-69      1.000000  0.000000
    user-1-7       0.424242  0.575758
    user-1-72      0.476190  0.523810
    user-1-76      0.346537  0.653463
    user-1-78      0.085106  0.914894
    user-1-8       0.863014  0.136986
    user-1-83      0.250000  0.750000
    user-1-9       0.311193  0.688807
    user-1-False   0.884488  0.115512
    


```python
print(pd.crosstab(data['source'], data['is_successful'], normalize='index'))
```

    is_successful                                              0         1
    source                                                                
    150ml Translucent Purple PET Boston Round, 20/4...  0.500000  0.500000
    30ml, 50ml and 100ml Airless 5k                     1.000000  0.000000
    45125 and 3829T-300T                                0.000000  1.000000
    Abandoned Basket                                    0.000000  1.000000
    Already a Customer                                  0.612005  0.387995
    ...                                                      ...       ...
    Website/Zendesk                                     0.575000  0.425000
    ♻️  Upgrade your Roll-on! NEW Aluminium Bottle ...  1.000000  0.000000
    👀 Low MOQ, High-Quality Aluminium Bottle Printi...  1.000000  0.000000
    👀 Low MOQ, High-Quality Aluminium Bottle Printi...  0.000000  1.000000
    💃Shake Things Up! With Our Eco-Friendly Alumini...  0.000000  1.000000
    
    [66 rows x 2 columns]
    

## ДОСЛІДЖЕННЯ РОЗПОДІЛУ

*Перевірка на асиметрію*:


```python
print(data.skew(numeric_only=True))
```

    is_successful                    -0.542771
    order_amount                     63.597978
    order_messages                    5.475379
    order_changes                     8.201835
    partner_success_rate             -0.708427
    partner_total_orders              3.626058
    partner_order_age_days            0.570445
    partner_avg_amount               92.806970
    partner_success_avg_amount       18.085258
    partner_fail_avg_amount          46.378813
    partner_total_messages            3.568156
    partner_success_avg_messages      1.976863
    partner_fail_avg_messages         4.583495
    partner_avg_changes               4.765490
    partner_success_avg_changes       2.553038
    partner_fail_avg_changes          7.842901
    quarter                           0.015067
    hour_of_day                       0.331366
    order_lines_count               175.787705
    discount_total                  113.972894
    dtype: float64
    

### Статистичні тести нормальності розподілу
**Тест Шапіро-Уілка** (scipy.stats.shapiro)
- Дуже точний, але для малих вибірок (до 5000 записів).

**Тест Д'Агостіно-Кільмейра** (scipy.stats.normaltest)
- Підходить для великих вибірок.

**Тест Андерсона-Дарлінга** (scipy.stats.anderson)
- Ще один строгий тест для нормальності.


```python
from scipy.stats import normaltest, anderson
```


```python
# Вибираємо тільки числові колонки
numeric_cols = data.select_dtypes(include=[np.number]).columns

# Створюємо результуючу таблицю
results = []

for col in numeric_cols:
    col_data = data[col]

    # Тест Д'Агостіно-Кільмейра
    stat_dagostino, p_dagostino = normaltest(col_data)

    # Тест Андерсона-Дарлінга
    result_anderson = anderson(col_data)
    stat_anderson = result_anderson.statistic
    critical_anderson = result_anderson.critical_values[2]  # поріг для рівня значущості 5%

    # Оцінка нормальності
    is_normal_dagostino = p_dagostino > 0.05
    is_normal_anderson = stat_anderson < critical_anderson

    results.append({
        'column': col,
        'dagostino_stat': stat_dagostino,
        'dagostino_p': p_dagostino,
        'dagostino_is_normal': is_normal_dagostino,
        'anderson_stat': stat_anderson,
        'anderson_critical_5%': critical_anderson,
        'anderson_is_normal': is_normal_anderson
    })

# Перетворюємо в DataFrame для красивого вигляду
normality_results = pd.DataFrame(results)
```


```python
print(normality_results)
```

                              column  dagostino_stat  dagostino_p  \
    0                  is_successful    3.747188e+05          0.0   
    1                   order_amount    2.970682e+05          0.0   
    2                 order_messages    9.640125e+04          0.0   
    3                  order_changes    1.278144e+05          0.0   
    4           partner_success_rate    8.396103e+03          0.0   
    5           partner_total_orders    6.608209e+04          0.0   
    6         partner_order_age_days    1.044213e+04          0.0   
    7             partner_avg_amount    3.370831e+05          0.0   
    8     partner_success_avg_amount    1.818324e+05          0.0   
    9        partner_fail_avg_amount    2.666633e+05          0.0   
    10        partner_total_messages    6.488780e+04          0.0   
    11  partner_success_avg_messages    4.625269e+04          0.0   
    12     partner_fail_avg_messages    8.721932e+04          0.0   
    13           partner_avg_changes    9.553359e+04          0.0   
    14   partner_success_avg_changes    5.739677e+04          0.0   
    15      partner_fail_avg_changes    1.224070e+05          0.0   
    16                       quarter    2.922516e+06          0.0   
    17                   hour_of_day    1.920232e+03          0.0   
    18             order_lines_count    4.092148e+05          0.0   
    19                discount_total    3.590099e+05          0.0   
    
        dagostino_is_normal  anderson_stat  anderson_critical_5%  \
    0                 False   17025.011072                 0.787   
    1                 False   26444.435265                 0.787   
    2                 False    7509.998823                 0.787   
    3                 False    7624.119674                 0.787   
    4                 False    2632.940401                 0.787   
    5                 False   11546.879620                 0.787   
    6                 False    1867.569814                 0.787   
    7                 False   22123.840104                 0.787   
    8                 False   13989.165778                 0.787   
    9                 False   21310.127781                 0.787   
    10                False   11645.300877                 0.787   
    11                False    2412.202000                 0.787   
    12                False    4743.869101                 0.787   
    13                False    2325.177258                 0.787   
    14                False    1932.346219                 0.787   
    15                False    5489.818281                 0.787   
    16                False    4298.009860                 0.787   
    17                False     945.511171                 0.787   
    18                False   14035.588981                 0.787   
    19                False   33429.633908                 0.787   
    
        anderson_is_normal  
    0                False  
    1                False  
    2                False  
    3                False  
    4                False  
    5                False  
    6                False  
    7                False  
    8                False  
    9                False  
    10               False  
    11               False  
    12               False  
    13               False  
    14               False  
    15               False  
    16               False  
    17               False  
    18               False  
    19               False  
    

Поояснення показників:

| Показник           | Що означає                                                               |
| :----------------- | :----------------------------------------------------------------------- |
| dagostino\_stat    | Статистика тесту Д’Агостіно-Кільмейра (скошеність + ексцес).               |
| dagostino\_p       | P-значення тесту Д’Агостіно-Кільмейра.                                   |
| dagostino\_is\_normal | Чи приймаємо нормальність за результатами Д’Агостіно-Кільмейра?         |
| anderson\_stat     | Статистика тесту Андерсона-Дарлінга.                                     |
| anderson\_critical\_5% | Критичне значення тесту Андерсона-Дарлінга при рівні 5%.                |
| anderson\_is\_normal | Чи приймаємо нормальність за результатами Андерсона-Дарлінга?           |

**Висновок:** *всі данні мають не нормальний розподіл*

### ДОСЛІДЖЕННЯ ЧИСЛОВИХ СТАТИСТИК


```python
def numeric_statistic(data):
    # 1. Виділяємо тільки числові колонки
    numeric_columns = data.select_dtypes(include=['number']).columns

    # 2. Створюємо список для зберігання результатів
    stats_list = []

    # 3. Обходимо всі числові колонки
    for col in numeric_columns:
        col_min = data[col].min()
        col_max = data[col].max()
        col_mean = data[col].mean()
        col_median = data[col].median()
        # mode() може повертати кілька значень, беремо перше
        col_mode = data[col].mode().iloc[0] if not data[col].mode().empty else None

        stats_list.append({
            'column': col,
            'min': col_min,
            'max': col_max,
            'mean': col_mean,
            'median': col_median,
            'mode': col_mode
        })

    # 4. Створюємо фінальний DataFrame
    stats_df = pd.DataFrame(stats_list)
    return stats_df
```

#### Додавання стовпця 'create_date_months'


```python
data['create_date'] = pd.to_datetime(data['create_date'])  # Перетворення стовпця 'create_date' у формат datetime
min_date = data['create_date'].min()  # Знаходимо найпершу дату
data.loc[:, 'create_date_months'] = ((data['create_date'] - min_date).dt.days / 30).astype(
    int)  # Обчислюємо різницю в місяцях, перетворюємо у int
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 86794 entries, 0 to 86793
    Data columns (total 26 columns):
     #   Column                        Non-Null Count  Dtype         
    ---  ------                        --------------  -----         
     0   is_successful                 86794 non-null  int64         
     1   create_date                   86794 non-null  datetime64[ns]
     2   order_amount                  86794 non-null  float64       
     3   order_messages                86794 non-null  int64         
     4   order_changes                 86794 non-null  int64         
     5   partner_success_rate          86794 non-null  float64       
     6   partner_total_orders          86794 non-null  int64         
     7   partner_order_age_days        86794 non-null  int64         
     8   partner_avg_amount            86794 non-null  float64       
     9   partner_success_avg_amount    86794 non-null  float64       
     10  partner_fail_avg_amount       86794 non-null  float64       
     11  partner_total_messages        86794 non-null  int64         
     12  partner_success_avg_messages  86794 non-null  float64       
     13  partner_fail_avg_messages     86794 non-null  float64       
     14  partner_avg_changes           86794 non-null  float64       
     15  partner_success_avg_changes   86794 non-null  float64       
     16  partner_fail_avg_changes      86794 non-null  float64       
     17  day_of_week                   86794 non-null  object        
     18  month                         86794 non-null  object        
     19  quarter                       86794 non-null  int64         
     20  hour_of_day                   86794 non-null  int64         
     21  order_lines_count             86794 non-null  int64         
     22  discount_total                86794 non-null  float64       
     23  salesperson                   86794 non-null  object        
     24  source                        86794 non-null  object        
     25  create_date_months            86794 non-null  int64         
    dtypes: datetime64[ns](1), float64(11), int64(10), object(4)
    memory usage: 17.2+ MB
    


```python
num_stat = numeric_statistic(data)
```


```python
print(num_stat)
```

                              column     min        max         mean       median  \
    0                  is_successful     0.0        1.0     0.630954     1.000000   
    1                   order_amount -6242.4  4140000.0  3824.406985   615.985000   
    2                 order_messages     1.0      282.0     9.408542     7.000000   
    3                  order_changes     0.0      274.0     3.904521     3.000000   
    4           partner_success_rate     0.0      100.0    59.600411    66.666667   
    5           partner_total_orders     0.0     1307.0    83.696258    26.000000   
    6         partner_order_age_days     0.0     2685.0   899.232136   764.000000   
    7             partner_avg_amount     0.0  2700000.0  2914.208745  1127.518158   
    8     partner_success_avg_amount     0.0   119310.0  1597.019479   933.960000   
    9        partner_fail_avg_amount     0.0  2700000.0  4112.985430   821.600000   
    10        partner_total_messages     0.0    11481.0   758.382688   240.000000   
    11  partner_success_avg_messages     0.0      123.5     9.556764     9.474948   
    12     partner_fail_avg_messages     0.0      143.0     5.705881     5.666667   
    13           partner_avg_changes     0.0      137.5     4.060877     3.969032   
    14   partner_success_avg_changes     0.0       57.0     4.409310     4.333333   
    15      partner_fail_avg_changes     0.0      137.5     3.058853     2.958333   
    16                       quarter     1.0        4.0     2.472821     2.000000   
    17                   hour_of_day     0.0       23.0    10.937000    11.000000   
    18             order_lines_count     0.0     1304.0     3.369645     3.000000   
    19                discount_total     0.0      390.0     0.039169     0.000000   
    20            create_date_months     0.0       90.0    43.099028    41.000000   
    
        mode  
    0    1.0  
    1    0.0  
    2    6.0  
    3    2.0  
    4    0.0  
    5    0.0  
    6    0.0  
    7    0.0  
    8    0.0  
    9    0.0  
    10   0.0  
    11   0.0  
    12   0.0  
    13   0.0  
    14   0.0  
    15   0.0  
    16   3.0  
    17  10.0  
    18   2.0  
    19   0.0  
    20  34.0  
    

#### Заміна від'ємних на 0


```python
# Виділяємо тільки числові колонки
numeric_columns = data.select_dtypes(include=['number']).columns

# Заміна від'ємних значень на 0 у числових стовпцях
for col in numeric_columns:
    data[col] = data[col].apply(lambda x: max(0, x))  # Застосовуємо функцію max(0, x) до кожного елемента стовпця
```

####  Виключення даних за 2025 рік


```python
data = data[data['create_date'].dt.year < 2025].copy()  # Фільтруємо рядки, залишаючи лише ті, де рік менш як 2025
```

#### Нормалізація `partner_success_rate`


```python
data['partner_success_rate'] = data['partner_success_rate'] / 100
```

### МАСШТАБУВАННЯ


```python
# Список колонок, які потрібно масштабувати
columns_to_scale = [
    'order_amount',
    'order_messages',
    'order_changes',
    'partner_total_orders',
    'partner_order_age_days',
    'partner_avg_amount',
    'partner_success_avg_amount',
    'partner_fail_avg_amount',
    'partner_total_messages',
    'partner_success_avg_messages',
    'partner_fail_avg_messages',
    'partner_avg_changes',
    'partner_success_avg_changes',
    'partner_fail_avg_changes',
    'order_lines_count',
    'discount_total',
    'create_date_months'
]
```


```python
from sklearn.preprocessing import RobustScaler
```


```python
# Ініціалізуємо RobustScaler
scaler = RobustScaler()

# Масштабуємо тільки числові дані
scaled_array = scaler.fit_transform(data[columns_to_scale])
```


```python
# Перетворюємо назад у DataFrame
scaled_df = pd.DataFrame(scaled_array, columns=[col for col in columns_to_scale])
```


```python
# Замінюємо оригінальні колонки в data масштабованими значеннями
data[columns_to_scale] = scaled_df[columns_to_scale]
```


```python
print(data)
```

           is_successful                create_date  order_amount  order_messages  \
    0                  1 2017-07-29 07:48:26.812523      2.421132        3.000000   
    1                  1 2017-07-29 07:54:09.954757      0.136770        0.500000   
    2                  1 2017-07-29 08:04:13.162858      1.256132        0.000000   
    3                  1 2017-07-29 08:11:38.086709      0.002801        0.500000   
    4                  1 2017-07-29 08:15:05.548616      0.103312       -0.166667   
    ...              ...                        ...           ...             ...   
    86787              1 2024-12-09 10:22:13.166600     -0.215944       -0.833333   
    86788              1 2024-12-10 11:07:58.049169     -0.266079       -0.333333   
    86789              1 2024-12-11 11:09:57.124395     -0.322561       -0.333333   
    86790              0 2024-12-16 08:38:35.387458      0.042648       -0.833333   
    86791              1 2024-12-19 13:59:08.130686     -0.254727       -0.833333   
    
           order_changes  partner_success_rate  partner_total_orders  \
    0                9.5              0.000000             -0.305882   
    1                1.0              0.000000             -0.305882   
    2                0.5              0.000000             -0.305882   
    3                1.5              0.000000             -0.305882   
    4                0.0              0.000000             -0.305882   
    ...              ...                   ...                   ...   
    86787           -1.0              0.816514              0.976471   
    86788            0.5              0.000000             -0.305882   
    86789           -0.5              0.111111             -0.200000   
    86790           -1.5              1.000000             -0.023529   
    86791           -1.0              0.000000             -0.305882   
    
           partner_order_age_days  partner_avg_amount  partner_success_avg_amount  \
    0                   -0.630884           -0.544190                   -0.687509   
    1                   -0.630884           -0.544190                   -0.687509   
    2                   -0.630884           -0.544190                   -0.687509   
    3                   -0.630884           -0.544190                   -0.687509   
    4                   -0.630884           -0.544190                   -0.687509   
    ...                       ...                 ...                         ...   
    86787                1.586292            2.705002                    3.408798   
    86788               -0.630884           -0.544190                   -0.687509   
    86789               -0.583815           33.907794                   -0.687509   
    86790               -0.009909           -0.288283                   -0.297169   
    86791               -0.630884           -0.544190                   -0.687509   
    
           ...  partner_fail_avg_changes  day_of_week     month  quarter  \
    0      ...                    -1.775     Saturday      July        3   
    1      ...                    -1.775     Saturday      July        3   
    2      ...                    -1.775     Saturday      July        3   
    3      ...                    -1.775     Saturday      July        3   
    4      ...                    -1.775     Saturday      July        3   
    ...    ...                       ...          ...       ...      ...   
    86787  ...                     1.495       Monday  December        4   
    86788  ...                    -1.775      Tuesday  December        4   
    86789  ...                     0.250    Wednesday  December        4   
    86790  ...                    -1.775       Monday  December        4   
    86791  ...                    -1.775     Thursday  December        4   
    
           hour_of_day  order_lines_count  discount_total salesperson source  \
    0                7                1.5             0.0   user-1-76  False   
    1                7                0.0             0.0   user-1-76  False   
    2                8                0.5             0.0    user-1-9  False   
    3                8                0.5             0.0    user-1-2  False   
    4                8                0.0             0.0    user-1-9  False   
    ...            ...                ...             ...         ...    ...   
    86787           10               -0.5             0.0   user-1-10  False   
    86788           11               -1.0             0.0   user-1-39  False   
    86789           11               -1.0             0.0   user-1-49  False   
    86790            8                0.0             0.0  user-1-113  False   
    86791           13               -1.0             0.0    user-1-2  False   
    
           create_date_months  
    0               -1.000000  
    1               -1.000000  
    2               -1.000000  
    3               -1.000000  
    4               -1.000000  
    ...                   ...  
    86787            1.170732  
    86788            1.170732  
    86789            1.170732  
    86790            1.170732  
    86791            1.195122  
    
    [86792 rows x 26 columns]
    

# ДОСЛІДЖЕННЯ КАТЕГОРІАЛЬНИХ

## `salesperson` та `source`

1. Унікальні значення


```python
data['salesperson'].unique()
```




    array(['user-1-76', 'user-1-9', 'user-1-2', 'user-1-49', 'user-1-10',
           'user-1-142', 'user-1-7', 'user-1-False', 'user-1-8', 'user-1-83',
           'user-1-100', 'user-1-54', 'user-1-69', 'user-1-113', 'user-1-72',
           'user-1-14', 'user-1-78', 'user-1-67', 'user-1-63', 'user-1-19',
           'user-1-1066', 'user-1-1414', 'user-1-1366', 'user-1-11',
           'user-1-4', 'user-1-56', 'user-1-1451', 'user-1-1465',
           'user-1-1582', 'user-1-47', 'user-1-39'], dtype=object)




```python
data['source'].unique()
```




    array(['False', 'Referral – Supplier – GPP',
           'Referral – Supplier – Grams', 'Google / Search engine',
           'Lead Generation', 'Already a Customer',
           'Proactive - Exhibition 11-12/09/2019', 'PI 2020',
           'Hand-San Enquiries', 'NOT ON PIPELINE', 'LOST ENQUIRY',
           'Mixed Hand-san', 'Website/Zendesk',
           '150ml Translucent Purple PET Boston Round, 20/410 Neck + UV Inhibitor (code TBC)',
           '30ml, 50ml and 100ml Airless 5k', 'Webpackaging',
           '45125 and 3829T-300T', 'Customer FU Call / F2F Visit', 'Facebook',
           'Email', 'LinkedIn', 'Referral - Supplier - Alpha',
           'Referral – Existing Customer', 'SalesBond', 'Past Enquiry',
           'Referral – Supplier – Non Core', 'Proactive – Telemarketing',
           'Sample Request', 'Referral – Supplier – RMP',
           'Website Contact Form', 'Enquire Now Form', 'Sample Request Form',
           'Exhibition - Packaging Innovations 2023', 'Proactive - Other',
           'Lapsed Customer', 'Brochure Form',
           'As Simple As... VMS\xa0🤸\u200d♀️ 🤸\u200d♂️ 2023-04-05 16:30:21',
           '💃Shake Things Up! With Our Eco-Friendly Aluminium Shaker Can! 2023-04-24 16:30:21',
           'Powder, Perfectly Packaged 2023-04-14 16:30:27',
           'Hair & Skin From Within 2023-05-15 16:30:23',
           '♻️  Upgrade your Roll-on! NEW Aluminium Bottle 2023-05-11 16:30:23',
           'Product Request Form',
           '👀 Low MOQ, High-Quality Aluminium Bottle Printing 2023-04-14 16:30:22',
           'Exhibition - Spirit of Christmas', 'BCMPA',
           'Inbound Phone Enquiry ',
           'Aluminium Sifter Cap ♻️ 2023-11-21 16:30:23',
           'Clear PET -  Is it the one for you? 2023-11-10 13:10:18',
           'Sales Inbox', 'Referral – Partner (Cambrian/Richmond etc)',
           'Think Sustainably - Post-Consumer Recycled PET ♻️ 2024-02-05 16:30:38',
           'PI 2024',
           '👀 Low MOQ, High-Quality Aluminium Bottle Printing 2023-04-14 16:30:24',
           'Aluminium Roll-On Cans 2023-04-28 16:30:32',
           'Referral – Supplier – Pretium', 'Lapsed Customer Lead',
           'Unbeatable Pair: Aluminium Bottles & Lotion Pumps 👏 2024-04-22 16:30:23',
           'Referral - Supplier - Plastikse',
           'Bespoke Pouch Packaging Solutions ✨ 2024-04-22 12:30:19',
           'Custom Packaging Boxes', 'Aluminium Pill Jars',
           'Pouches Sample Pack Form',
           'Coming Soon! 100% PE Recyclable Disc Tops 2024-08-19 16:30:29',
           'Imbibe London 2023', 'Abandoned Basket',
           'In Stock: 100% PE Disc Tops ♻️ – Ready to Ship! 2024-10-23 08:30:39'],
          dtype=object)



2. Розподіл унікальних значень


```python
print(data['salesperson'].value_counts())
```

    salesperson
    user-1-2        25890
    user-1-9        21424
    user-1-76       14036
    user-1-49        8267
    user-1-10        7927
    user-1-113       3623
    user-1-7         1881
    user-1-142       1317
    user-1-1414      1134
    user-1-14         378
    user-1-False      303
    user-1-78         188
    user-1-67         144
    user-1-8           73
    user-1-47          30
    user-1-1366        28
    user-1-19          24
    user-1-72          21
    user-1-11          21
    user-1-83          20
    user-1-1066        18
    user-1-100         15
    user-1-4            9
    user-1-1582         6
    user-1-54           5
    user-1-69           3
    user-1-39           2
    user-1-56           2
    user-1-63           1
    user-1-1451         1
    user-1-1465         1
    Name: count, dtype: int64
    


```python
print(data['source'].value_counts())
```

    source
    False                                                                  80097
    Already a Customer                                                      2299
    NOT ON PIPELINE                                                         1908
    Hand-San Enquiries                                                       618
    Product Request Form                                                     401
                                                                           ...  
    Bespoke Pouch Packaging Solutions ✨ 2024-04-22 12:30:19                    1
    Coming Soon! 100% PE Recyclable Disc Tops 2024-08-19 16:30:29              1
    Imbibe London 2023                                                         1
    Abandoned Basket                                                           1
    In Stock: 100% PE Disc Tops ♻️ – Ready to Ship! 2024-10-23 08:30:39        1
    Name: count, Length: 66, dtype: int64
    

3. Розподіл `salesperson` за `is_successful`


```python
salesperson_success = data.groupby('salesperson')['is_successful'].value_counts(normalize=True).unstack(fill_value=0)
```


```python
print(salesperson_success)
```

    is_successful         0         1
    salesperson                      
    user-1-10      0.294941  0.705059
    user-1-100     0.400000  0.600000
    user-1-1066    0.500000  0.500000
    user-1-11      0.047619  0.952381
    user-1-113     0.299200  0.700800
    user-1-1366    0.214286  0.785714
    user-1-14      0.150794  0.849206
    user-1-1414    0.312169  0.687831
    user-1-142     0.290053  0.709947
    user-1-1451    0.000000  1.000000
    user-1-1465    1.000000  0.000000
    user-1-1582    0.000000  1.000000
    user-1-19      0.291667  0.708333
    user-1-2       0.459173  0.540827
    user-1-39      0.500000  0.500000
    user-1-4       0.111111  0.888889
    user-1-47      0.233333  0.766667
    user-1-49      0.384541  0.615459
    user-1-54      0.400000  0.600000
    user-1-56      0.500000  0.500000
    user-1-63      0.000000  1.000000
    user-1-67      0.090278  0.909722
    user-1-69      1.000000  0.000000
    user-1-7       0.424242  0.575758
    user-1-72      0.476190  0.523810
    user-1-76      0.346537  0.653463
    user-1-78      0.085106  0.914894
    user-1-8       0.863014  0.136986
    user-1-83      0.250000  0.750000
    user-1-9       0.311193  0.688807
    user-1-False   0.884488  0.115512
    

4. Розподіл `source` за `is_successful`


```python
source_success = data.groupby('source')['is_successful'].value_counts(normalize=True).unstack(fill_value=0)
```


```python
print(source_success)
```

    is_successful                                              0         1
    source                                                                
    150ml Translucent Purple PET Boston Round, 20/4...  0.500000  0.500000
    30ml, 50ml and 100ml Airless 5k                     1.000000  0.000000
    45125 and 3829T-300T                                0.000000  1.000000
    Abandoned Basket                                    0.000000  1.000000
    Already a Customer                                  0.612005  0.387995
    ...                                                      ...       ...
    Website/Zendesk                                     0.575000  0.425000
    ♻️  Upgrade your Roll-on! NEW Aluminium Bottle ...  1.000000  0.000000
    👀 Low MOQ, High-Quality Aluminium Bottle Printi...  1.000000  0.000000
    👀 Low MOQ, High-Quality Aluminium Bottle Printi...  0.000000  1.000000
    💃Shake Things Up! With Our Eco-Friendly Alumini...  0.000000  1.000000
    
    [66 rows x 2 columns]
    

## Кодуємо`salesperson` та `source`


```python
# Частотне кодування для 'salesperson'
salesperson_freq = data['salesperson'].value_counts(normalize=True)
data['salesperson'] = data['salesperson'].map(salesperson_freq)
```


```python
# Те саме для 'source'
source_freq = data['source'].value_counts(normalize=True)
data['source'] = data['source'].map(source_freq)
```


```python
data['salesperson'].unique()
```




    array([1.61719974e-01, 2.46843027e-01, 2.98299382e-01, 9.52507144e-02,
           9.13333026e-02, 1.51742096e-02, 2.16725044e-02, 3.49110517e-03,
           8.41091345e-04, 2.30435985e-04, 1.72826989e-04, 5.76089962e-05,
           3.45653977e-05, 4.17434787e-02, 2.41957784e-04, 4.35524011e-03,
           2.16609826e-03, 1.65913909e-03, 1.15217992e-05, 2.76523182e-04,
           2.07392386e-04, 1.30657203e-02, 3.22610379e-04, 1.03696193e-04,
           2.30435985e-05, 6.91307955e-05, 3.45653977e-04])




```python
data['source'].unique()
```




    array([9.22861554e-01, 1.03696193e-04, 1.15217992e-05, 4.14784773e-04,
           5.76089962e-05, 2.64886165e-02, 1.60153009e-03, 7.12047193e-03,
           2.19835930e-02, 8.18047746e-04, 3.45653977e-05, 1.84348788e-03,
           2.30435985e-05, 3.34132178e-04, 1.95870587e-04, 1.01391833e-03,
           9.21743940e-05, 4.60871970e-05, 5.18480966e-04, 8.06525947e-05,
           6.79786155e-04, 1.65913909e-03, 2.76523182e-04, 1.53239930e-03,
           7.71960549e-04, 1.61305189e-04, 1.84348788e-04, 4.62024150e-03,
           1.07152733e-03, 1.14065813e-03, 1.72826989e-04, 9.79352936e-04,
           6.91307955e-05, 1.15217992e-04])



## Дослідження ознак дати та часу


```python
# Дослідження колонки 'day_of_week'
print("\nРозподіл за днями тижня:")
print(data['day_of_week'].value_counts().sort_index())
```

    
    Розподіл за днями тижня:
    day_of_week
    Friday       15188
    Monday       16307
    Saturday       139
    Sunday         122
    Thursday     17679
    Tuesday      18785
    Wednesday    18572
    Name: count, dtype: int64
    


```python
# Дослідження колонки 'month'
print("\nРозподіл за місяцями:")
print(data['month'].value_counts().sort_index())
```

    
    Розподіл за місяцями:
    month
    April        7265
    August       7740
    December     4694
    February     7052
    January      7176
    July         7212
    June         7208
    March        8093
    May          6850
    November     7349
    October      8170
    September    7983
    Name: count, dtype: int64
    


```python
# Дослідження колонки 'quarter'
print("\nРозподіл за кварталами:")
print(data['quarter'].value_counts().sort_index())
```

    
    Розподіл за кварталами:
    quarter
    1    22321
    2    21323
    3    22935
    4    20213
    Name: count, dtype: int64
    


```python
# Дослідження колонки 'hour_of_day'
print("\nРозподіл за годинами:")
print(data['hour_of_day'].value_counts().sort_index())
```

    
    Розподіл за годинами:
    hour_of_day
    0         3
    1         2
    2         2
    3         2
    4        25
    5       325
    6      2526
    7      7768
    8     10245
    9     10072
    10    10675
    11     8801
    12     8150
    13     9004
    14     8811
    15     6358
    16     2709
    17      488
    18      110
    19      147
    20      233
    21      243
    22       79
    23       14
    Name: count, dtype: int64
    


```python
# Створення мапінгів для днів тижня та місяців
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
```


```python
# Застосування мапінгів до відповідних колонок
data['day_of_week'] = data['day_of_week'].map(day_mapping)
data['month'] = data['month'].map(month_mapping)
```


```python
# Ініціалізуй енкодер для циклічних ознак
cyclical_encoder = CyclicalFeatures(variables=['day_of_week', 'month', 'quarter', 'hour_of_day'],
                                     drop_original=True)
```


```python
# Навчай та трансформуй дані
data = cyclical_encoder.fit_transform(data)
```


```python
print(data)
```

           is_successful                create_date  order_amount  order_messages  \
    0                  1 2017-07-29 07:48:26.812523      2.421132        3.000000   
    1                  1 2017-07-29 07:54:09.954757      0.136770        0.500000   
    2                  1 2017-07-29 08:04:13.162858      1.256132        0.000000   
    3                  1 2017-07-29 08:11:38.086709      0.002801        0.500000   
    4                  1 2017-07-29 08:15:05.548616      0.103312       -0.166667   
    ...              ...                        ...           ...             ...   
    86787              1 2024-12-09 10:22:13.166600     -0.215944       -0.833333   
    86788              1 2024-12-10 11:07:58.049169     -0.266079       -0.333333   
    86789              1 2024-12-11 11:09:57.124395     -0.322561       -0.333333   
    86790              0 2024-12-16 08:38:35.387458      0.042648       -0.833333   
    86791              1 2024-12-19 13:59:08.130686     -0.254727       -0.833333   
    
           order_changes  partner_success_rate  partner_total_orders  \
    0                9.5              0.000000             -0.305882   
    1                1.0              0.000000             -0.305882   
    2                0.5              0.000000             -0.305882   
    3                1.5              0.000000             -0.305882   
    4                0.0              0.000000             -0.305882   
    ...              ...                   ...                   ...   
    86787           -1.0              0.816514              0.976471   
    86788            0.5              0.000000             -0.305882   
    86789           -0.5              0.111111             -0.200000   
    86790           -1.5              1.000000             -0.023529   
    86791           -1.0              0.000000             -0.305882   
    
           partner_order_age_days  partner_avg_amount  partner_success_avg_amount  \
    0                   -0.630884           -0.544190                   -0.687509   
    1                   -0.630884           -0.544190                   -0.687509   
    2                   -0.630884           -0.544190                   -0.687509   
    3                   -0.630884           -0.544190                   -0.687509   
    4                   -0.630884           -0.544190                   -0.687509   
    ...                       ...                 ...                         ...   
    86787                1.586292            2.705002                    3.408798   
    86788               -0.630884           -0.544190                   -0.687509   
    86789               -0.583815           33.907794                   -0.687509   
    86790               -0.009909           -0.288283                   -0.297169   
    86791               -0.630884           -0.544190                   -0.687509   
    
           ...    source  create_date_months  day_of_week_sin  day_of_week_cos  \
    0      ...  0.922862           -1.000000        -0.781831         0.623490   
    1      ...  0.922862           -1.000000        -0.781831         0.623490   
    2      ...  0.922862           -1.000000        -0.781831         0.623490   
    3      ...  0.922862           -1.000000        -0.781831         0.623490   
    4      ...  0.922862           -1.000000        -0.781831         0.623490   
    ...    ...       ...                 ...              ...              ...   
    86787  ...  0.922862            1.170732         0.781831         0.623490   
    86788  ...  0.922862            1.170732         0.974928        -0.222521   
    86789  ...  0.922862            1.170732         0.433884        -0.900969   
    86790  ...  0.922862            1.170732         0.781831         0.623490   
    86791  ...  0.922862            1.195122        -0.433884        -0.900969   
    
              month_sin  month_cos   quarter_sin   quarter_cos  hour_of_day_sin  \
    0     -5.000000e-01  -0.866025 -1.000000e+00 -1.836970e-16         0.942261   
    1     -5.000000e-01  -0.866025 -1.000000e+00 -1.836970e-16         0.942261   
    2     -5.000000e-01  -0.866025 -1.000000e+00 -1.836970e-16         0.816970   
    3     -5.000000e-01  -0.866025 -1.000000e+00 -1.836970e-16         0.816970   
    4     -5.000000e-01  -0.866025 -1.000000e+00 -1.836970e-16         0.816970   
    ...             ...        ...           ...           ...              ...   
    86787 -2.449294e-16   1.000000 -2.449294e-16  1.000000e+00         0.398401   
    86788 -2.449294e-16   1.000000 -2.449294e-16  1.000000e+00         0.136167   
    86789 -2.449294e-16   1.000000 -2.449294e-16  1.000000e+00         0.136167   
    86790 -2.449294e-16   1.000000 -2.449294e-16  1.000000e+00         0.816970   
    86791 -2.449294e-16   1.000000 -2.449294e-16  1.000000e+00        -0.398401   
    
           hour_of_day_cos  
    0            -0.334880  
    1            -0.334880  
    2            -0.576680  
    3            -0.576680  
    4            -0.576680  
    ...                ...  
    86787        -0.917211  
    86788        -0.990686  
    86789        -0.990686  
    86790        -0.576680  
    86791        -0.917211  
    
    [86792 rows x 30 columns]
    


```python
print(data.head(10))
```

       is_successful                create_date  order_amount  order_messages  \
    0              1 2017-07-29 07:48:26.812523      2.421132        3.000000   
    1              1 2017-07-29 07:54:09.954757      0.136770        0.500000   
    2              1 2017-07-29 08:04:13.162858      1.256132        0.000000   
    3              1 2017-07-29 08:11:38.086709      0.002801        0.500000   
    4              1 2017-07-29 08:15:05.548616      0.103312       -0.166667   
    5              1 2017-07-29 08:19:38.625071      0.216778        2.500000   
    6              1 2017-07-29 08:29:50.487564      1.161392        0.500000   
    7              1 2017-07-29 08:40:21.407789     -0.144238        0.500000   
    8              1 2017-07-29 08:44:45.461359      0.487828        0.666667   
    9              1 2017-07-29 08:53:03.880701      0.767693        0.166667   
    
       order_changes  partner_success_rate  partner_total_orders  \
    0            9.5                   0.0             -0.305882   
    1            1.0                   0.0             -0.305882   
    2            0.5                   0.0             -0.305882   
    3            1.5                   0.0             -0.305882   
    4            0.0                   0.0             -0.305882   
    5            8.0                   0.0             -0.305882   
    6            2.0                   0.0             -0.305882   
    7            1.5                   0.0             -0.305882   
    8            2.5                   0.0             -0.305882   
    9            0.5                   0.0             -0.305882   
    
       partner_order_age_days  partner_avg_amount  partner_success_avg_amount  \
    0               -0.630884            -0.54419                   -0.687509   
    1               -0.630884            -0.54419                   -0.687509   
    2               -0.630884            -0.54419                   -0.687509   
    3               -0.630884            -0.54419                   -0.687509   
    4               -0.630884            -0.54419                   -0.687509   
    5               -0.630884            -0.54419                   -0.687509   
    6               -0.630884            -0.54419                   -0.687509   
    7               -0.630884            -0.54419                   -0.687509   
    8               -0.630884            -0.54419                   -0.687509   
    9               -0.630884            -0.54419                   -0.687509   
    
       ...    source  create_date_months  day_of_week_sin  day_of_week_cos  \
    0  ...  0.922862                -1.0        -0.781831          0.62349   
    1  ...  0.922862                -1.0        -0.781831          0.62349   
    2  ...  0.922862                -1.0        -0.781831          0.62349   
    3  ...  0.922862                -1.0        -0.781831          0.62349   
    4  ...  0.922862                -1.0        -0.781831          0.62349   
    5  ...  0.922862                -1.0        -0.781831          0.62349   
    6  ...  0.922862                -1.0        -0.781831          0.62349   
    7  ...  0.922862                -1.0        -0.781831          0.62349   
    8  ...  0.922862                -1.0        -0.781831          0.62349   
    9  ...  0.922862                -1.0        -0.781831          0.62349   
    
       month_sin  month_cos  quarter_sin   quarter_cos  hour_of_day_sin  \
    0       -0.5  -0.866025         -1.0 -1.836970e-16         0.942261   
    1       -0.5  -0.866025         -1.0 -1.836970e-16         0.942261   
    2       -0.5  -0.866025         -1.0 -1.836970e-16         0.816970   
    3       -0.5  -0.866025         -1.0 -1.836970e-16         0.816970   
    4       -0.5  -0.866025         -1.0 -1.836970e-16         0.816970   
    5       -0.5  -0.866025         -1.0 -1.836970e-16         0.816970   
    6       -0.5  -0.866025         -1.0 -1.836970e-16         0.816970   
    7       -0.5  -0.866025         -1.0 -1.836970e-16         0.816970   
    8       -0.5  -0.866025         -1.0 -1.836970e-16         0.816970   
    9       -0.5  -0.866025         -1.0 -1.836970e-16         0.816970   
    
       hour_of_day_cos  
    0         -0.33488  
    1         -0.33488  
    2         -0.57668  
    3         -0.57668  
    4         -0.57668  
    5         -0.57668  
    6         -0.57668  
    7         -0.57668  
    8         -0.57668  
    9         -0.57668  
    
    [10 rows x 30 columns]
    


```python
print(data.tail(10))
```

           is_successful                create_date  order_amount  order_messages  \
    86782              0 2024-11-26 09:17:07.654485     -0.322838       -1.000000   
    86783              1 2024-11-26 09:20:42.531614     -0.280036       -0.833333   
    86784              1 2024-11-26 09:23:06.451541     -0.280036       -0.833333   
    86785              0 2024-11-26 09:34:52.576840     -0.322838       -1.000000   
    86786              0 2024-12-04 15:14:51.956392     -0.322838       -1.000000   
    86787              1 2024-12-09 10:22:13.166600     -0.215944       -0.833333   
    86788              1 2024-12-10 11:07:58.049169     -0.266079       -0.333333   
    86789              1 2024-12-11 11:09:57.124395     -0.322561       -0.333333   
    86790              0 2024-12-16 08:38:35.387458      0.042648       -0.833333   
    86791              1 2024-12-19 13:59:08.130686     -0.254727       -0.833333   
    
           order_changes  partner_success_rate  partner_total_orders  \
    86782           -1.5              0.569231              1.223529   
    86783           -1.0              0.000000             -0.282353   
    86784           -1.0              0.333333             -0.270588   
    86785           -1.5              0.000000             -0.305882   
    86786           -1.5              0.581967              1.129412   
    86787           -1.0              0.816514              0.976471   
    86788            0.5              0.000000             -0.305882   
    86789           -0.5              0.111111             -0.200000   
    86790           -1.5              1.000000             -0.023529   
    86791           -1.0              0.000000             -0.305882   
    
           partner_order_age_days  partner_avg_amount  partner_success_avg_amount  \
    86782                1.562345            2.338429                    3.177613   
    86783               -0.256813           -0.544054                   -0.687509   
    86784               -0.256813           -0.530961                   -0.627390   
    86785               -0.630884           -0.544190                   -0.687509   
    86786                1.151115            1.605849                    1.572204   
    86787                1.586292            2.705002                    3.408798   
    86788               -0.630884           -0.544190                   -0.687509   
    86789               -0.583815           33.907794                   -0.687509   
    86790               -0.009909           -0.288283                   -0.297169   
    86791               -0.630884           -0.544190                   -0.687509   
    
           ...    source  create_date_months  day_of_week_sin  day_of_week_cos  \
    86782  ...  0.922862            1.170732         0.974928        -0.222521   
    86783  ...  0.922862            1.170732         0.974928        -0.222521   
    86784  ...  0.922862            1.170732         0.974928        -0.222521   
    86785  ...  0.922862            1.170732         0.974928        -0.222521   
    86786  ...  0.922862            1.170732         0.433884        -0.900969   
    86787  ...  0.922862            1.170732         0.781831         0.623490   
    86788  ...  0.922862            1.170732         0.974928        -0.222521   
    86789  ...  0.922862            1.170732         0.433884        -0.900969   
    86790  ...  0.922862            1.170732         0.781831         0.623490   
    86791  ...  0.922862            1.195122        -0.433884        -0.900969   
    
              month_sin  month_cos   quarter_sin  quarter_cos  hour_of_day_sin  \
    86782 -5.000000e-01   0.866025 -2.449294e-16          1.0         0.631088   
    86783 -5.000000e-01   0.866025 -2.449294e-16          1.0         0.631088   
    86784 -5.000000e-01   0.866025 -2.449294e-16          1.0         0.631088   
    86785 -5.000000e-01   0.866025 -2.449294e-16          1.0         0.631088   
    86786 -2.449294e-16   1.000000 -2.449294e-16          1.0        -0.816970   
    86787 -2.449294e-16   1.000000 -2.449294e-16          1.0         0.398401   
    86788 -2.449294e-16   1.000000 -2.449294e-16          1.0         0.136167   
    86789 -2.449294e-16   1.000000 -2.449294e-16          1.0         0.136167   
    86790 -2.449294e-16   1.000000 -2.449294e-16          1.0         0.816970   
    86791 -2.449294e-16   1.000000 -2.449294e-16          1.0        -0.398401   
    
           hour_of_day_cos  
    86782        -0.775711  
    86783        -0.775711  
    86784        -0.775711  
    86785        -0.775711  
    86786        -0.576680  
    86787        -0.917211  
    86788        -0.990686  
    86789        -0.990686  
    86790        -0.576680  
    86791        -0.917211  
    
    [10 rows x 30 columns]
    


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 86792 entries, 0 to 86791
    Data columns (total 30 columns):
     #   Column                        Non-Null Count  Dtype         
    ---  ------                        --------------  -----         
     0   is_successful                 86792 non-null  int64         
     1   create_date                   86792 non-null  datetime64[ns]
     2   order_amount                  86792 non-null  float64       
     3   order_messages                86792 non-null  float64       
     4   order_changes                 86792 non-null  float64       
     5   partner_success_rate          86792 non-null  float64       
     6   partner_total_orders          86792 non-null  float64       
     7   partner_order_age_days        86792 non-null  float64       
     8   partner_avg_amount            86792 non-null  float64       
     9   partner_success_avg_amount    86792 non-null  float64       
     10  partner_fail_avg_amount       86792 non-null  float64       
     11  partner_total_messages        86792 non-null  float64       
     12  partner_success_avg_messages  86792 non-null  float64       
     13  partner_fail_avg_messages     86792 non-null  float64       
     14  partner_avg_changes           86792 non-null  float64       
     15  partner_success_avg_changes   86792 non-null  float64       
     16  partner_fail_avg_changes      86792 non-null  float64       
     17  order_lines_count             86792 non-null  float64       
     18  discount_total                86792 non-null  float64       
     19  salesperson                   86792 non-null  float64       
     20  source                        86792 non-null  float64       
     21  create_date_months            86792 non-null  float64       
     22  day_of_week_sin               86792 non-null  float64       
     23  day_of_week_cos               86792 non-null  float64       
     24  month_sin                     86792 non-null  float64       
     25  month_cos                     86792 non-null  float64       
     26  quarter_sin                   86792 non-null  float64       
     27  quarter_cos                   86792 non-null  float64       
     28  hour_of_day_sin               86792 non-null  float64       
     29  hour_of_day_cos               86792 non-null  float64       
    dtypes: datetime64[ns](1), float64(28), int64(1)
    memory usage: 20.5 MB
    


```python
print(data.describe())
```

           is_successful                    create_date  order_amount  \
    count   86792.000000                          86792  86792.000000   
    mean        0.630945  2021-02-25 23:17:14.904852736      1.681594   
    min         0.000000     2017-07-29 07:48:26.812523     -0.322838   
    25%         0.000000  2019-06-26 08:24:19.036283648     -0.320737   
    50%         1.000000  2021-01-06 07:53:00.763869952      0.000000   
    75%         1.000000  2022-11-28 15:51:16.311046400      0.679263   
    max         1.000000     2024-12-19 13:59:08.130686   2169.420251   
    std         0.482552                            NaN     16.462440   
    
           order_messages  order_changes  partner_success_rate  \
    count    86792.000000   86792.000000          86792.000000   
    mean         0.401448       0.452288              0.595995   
    min         -1.000000      -1.500000              0.000000   
    25%         -0.333333      -0.500000              0.441860   
    50%          0.000000       0.000000              0.666667   
    75%          0.666667       0.500000              0.833333   
    max         45.833333     135.500000              1.000000   
    std          1.370445       1.918219              0.313057   
    
           partner_total_orders  partner_order_age_days  partner_avg_amount  \
    count          86792.000000            86792.000000        86792.000000   
    mean               0.678801                0.111678            0.862243   
    min               -0.305882               -0.630884           -0.544190   
    25%               -0.247059               -0.450041           -0.315511   
    50%                0.000000                0.000000            0.000000   
    75%                0.752941                0.549959            0.684489   
    max               15.070588                1.586292         1302.480880   
    std                1.776630                0.622749            6.727057   
    
           partner_success_avg_amount  ...        source  create_date_months  \
    count                86792.000000  ...  86792.000000        86792.000000   
    mean                     0.488117  ...      0.852949            0.051169   
    min                     -0.687509  ...      0.000012           -1.000000   
    25%                     -0.370389  ...      0.922862           -0.439024   
    50%                      0.000000  ...      0.922862            0.000000   
    75%                      0.629611  ...      0.922862            0.560976   
    max                     87.139277  ...      0.922862            1.195122   
    std                      2.622237  ...      0.241839            0.607934   
    
           day_of_week_sin  day_of_week_cos     month_sin     month_cos  \
    count     86792.000000     86792.000000  8.679200e+04  8.679200e+04   
    mean          0.190512        -0.343866 -1.770991e-02 -2.309579e-02   
    min          -0.974928        -0.900969 -1.000000e+00 -1.000000e+00   
    25%          -0.433884        -0.900969 -8.660254e-01 -5.000000e-01   
    50%           0.433884        -0.222521  1.224647e-16 -1.836970e-16   
    75%           0.781831        -0.222521  8.660254e-01  5.000000e-01   
    max           0.974928         1.000000  1.000000e+00  1.000000e+00   
    std           0.728159         0.561476  7.269542e-01  6.860773e-01   
    
            quarter_sin   quarter_cos  hour_of_day_sin  hour_of_day_cos  
    count  8.679200e+04  8.679200e+04     86792.000000     86792.000000  
    mean  -7.074385e-03 -1.278920e-02         0.130349        -0.716047  
    min   -1.000000e+00 -1.000000e+00        -0.997669        -0.990686  
    25%   -1.000000e+00 -1.836970e-16        -0.398401        -0.917211  
    50%    1.224647e-16 -1.836970e-16         0.136167        -0.775711  
    75%    1.000000e+00  6.123234e-17         0.631088        -0.576680  
    max    1.000000e+00  1.000000e+00         0.997669         1.000000  
    std    7.220710e-01  6.916729e-01         0.622364         0.288017  
    
    [8 rows x 30 columns]
    
