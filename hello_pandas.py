import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import string
import matplotlib.pyplot as plt
import matplotlib as mpl

def series_basic():
    print(pd.__version__)
    s = Series([1, 2, 3, 4, 5])
    print(s)

    s = Series(np.random.randint(0, 1000, 10))
    print(s)

    print("Use index not state from 0")
    s = Series([10, 20, 30, 40, 50],
                index=[100, 200, 300, 400, 500])
    print(s)
    print(s[200])

    print("Use non-number indexes")
    s = Series([10, 20, 30, 40, 50],
                index=list('abcde'))
    print(s)
    print(s['c'])

    print("Use duplicate indexes")
    s = Series([10, 20, 30, 40, 50],
                index=list('abcba'))
    print(s)
    print(s['a'])
    print(s[['a', 'b']])
    print(s[['a', 'b']].mean())

    print("Use numberic index")
    print(s.loc['c'])
    print(s.iloc[2])

    print(s.index)

    print("Excise:")
    s = Series(np.random.randint(0, 1000, 10))
    print(s)
    print(s[[0]])
    print(s.min())
    print(s.max())
    print(s[s<s.mean()])

    s = Series(np.random.rand(5), index=list('abcde'))
    print(s[list('ace')].sum())

def series_op():
    s1 = Series(np.arange(1, 6, 1)*10, index=list('abcde'))
    s2 = Series(np.arange(1, 6, 1)*100, index=list('abcde'))
    s3 = Series(np.arange(1, 6, 1)*100, index=list('abcba'))
    print(s1)
    print(s2)
    print(s3)
    print(s1+s1)
    print(s1.add(s2))
    print(s1)
    print(s1.add(s3))
    print(s1.add(s3, fill_value=0))

    print("Excise:")
    s1 = Series(np.random.rand(10)*1000, index=list(string.ascii_lowercase[:10]))
    s2 = Series(np.random.rand(10)*1000, index=list(string.ascii_lowercase[:5]*2))
    print(s1)
    print(s2)

    print(pd.concat([s1, s2]))

    print(s1[list('bcd')].add(s2[list('bcd')]))
    print(s1[list('bcd')].add(s2[list('bcd')], fill_value=0))
    print(s1[list('efg')].add(s2[list('efg')]))
    print(s1[list('efg')].add(s2[list('efg')], fill_value=0))

def series_adv():

    print("Ignore nan in pandas which is different from numpy")
    s = Series([10, 11, np.nan, 13, 14])
    print(s)
    print(s.sum())

    s = Series(np.random.randint(0, 10, 100))
    print(s)
    print(s.head(5))
    s = s.value_counts()
    print(s)
    print(s.head(5))

    print("Apply str.len to all items in Series")
    words = "this is some workds use by my machine learinnng classes".split()
    s = Series(words)
    print(s)
    print(s[s.str.len() > 4])

    print("Excise:")
    words = "this 30 is some 3 value 333ab from 333 de 111".split()
    s = Series(words)
    print(s)
    print(s[s.str.isdigit()].astype(np.float64).sum())
    words = '''The hardest and most complex part this installation the Jupyter Notebook, previously known as the IPython notebook. This is a modern, advanced, browser-based interactive Python shell. It allows you to experiment with Python code. I describe it as a laboratory for'''.split()

    s = Series(words)
    print(s.str.len().value_counts().head(3))

    s = Series(np.random.randint(0, 1000, 100))
    print(s.describe())

def pandas_plot():
    s = Series(np.random.randint(0, 30, 20))
    # in jupetor
    s.plot(title='hello')
    s.plot.bar()
    plt.show()
    df = pd.read_csv('files/taxi.csv', usecols=['passenger_count', 'trip_distance', 'total_amount'])
    #print(df.corr())

    from pandas.plotting import scatter_matrix
    scatter_matrix(df)
    plt.show()
    
def matplotlib_plot():
    s = Series(np.random.randint(0, 30, 20))
    plt.plot(s)
    plt.show()

    df = pd.read_csv('files/taxi.csv', usecols=['passenger_count', 'trip_distance', 'total_amount'])
    plt.plot(df)
    plt.show()


def seeborn_plot():
    # not installed
    import seeborn
    s = Series(np.random.randint(0, 30, 20))

def dataframe_basic():
    df = DataFrame([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    print(df)
    df = DataFrame([[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                    index=list('xyz'),
                    columns=list('abc'))
    print(df)

    df = DataFrame(np.arange(10, 100, 10).reshape(3, 3))
    print(df)
    df = DataFrame(np.random.randint(1, 1000, 25).reshape(5, 5))
    print(df)

    print("Column is Series")
    df = DataFrame([[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                    index=list('xyz'),
                    columns=list('abc'))
    print(df)
    print(df['a'])
    print(df.loc['x'])

    print("Excise:")
    df = DataFrame(np.random.randint(1, 1000, 16).reshape(4, 4),
                    index=list('wxyz'),
                    columns=list('abcd'))
    print(df)
    print(df['b'])
    print(df.loc['x'])
    print(df[['a', 'c']])
    print(df[['c', 'd']].loc[['y', 'z']])
    print(df[['c', 'd']].iloc[[2, 3]])

    # slice is apply to rows
    print(df['w':'y']) # char slice, 'y' is included
    print(df[1:3]) # number slice, 3 is not included


    print("Create dataframe with list of dictionaries")
    df = DataFrame([{'a':1, 'b':3, 'd':4},
                    {'a':'aa', 'b':'bb', 'd':'dd'},
                    {'a':11, 'b':22, 'c':33}],
                    index=list('xyz'))
    print(df)

    print("Create dataframe with dictionary of lists")
    df = DataFrame({'a':[1, 2, 3, 4],
                    'b':[10, 20, 30, 40],
                    'c':[100, 200, 300, 400]},
                    index=list('xyzh'))
    print(df)
    print("Add column")
    df['d'] = ['z', 'x', 'v', 'c']
    print(df)
    print("Add row")
    df.loc['k'] = ['f', 'f', 'k', 'k']
    print(df)

    df = DataFrame(np.random.randint(0, 1000, 16).reshape(4, 4),
                    index=list('wxyz'),
                    columns=list('abcd'))
    print(df)
    df['id'] = [1991, 1992, 2000, 2014]
    print(df)
    df.set_index('id', inplace=True)
    print(df)
    print(df.index)
    print(df.loc[[1992, 2014]])
    df.reset_index(inplace=True)
    print(df)
    print(df['id'])


def dataframe_op():
    df = DataFrame(np.random.randint(0, 100, 36).reshape(6, 6),
                    index=list('uvwxyz'),
                    columns=list('abcdef'))
    print(df)
    print(df<50)
    print(df['f'][df['e'] % 2 == 0])

    print("Excise:")
    print(df[df['a']<50])
    print(df[df['b']>df['b'].mean()])
    #print(df.loc[['v', 'w']][df['e']%2 == 0])
    print(df[df['e']%2 == 0])
    print(df[(df['c'] % 2 == 1) & (df['c'] > 20)][['a', 'b']].loc[['x', 'y']])

    #df.drop()


def dataframe_buildin():
    df = DataFrame(np.random.randint(1, 1000, 25).reshape(5, 5))
    print(df)
    print(df.info())
    print(df.info(memory_usage='deep'))
    print(df.describe())

def dataframe_file():
    df = DataFrame(np.random.randint(1, 1000, 25).reshape(5, 5),
                    index=list('abcde'),
                    columns=list('wszxv'))
    print(df)
    df.to_csv('mydf.csv', sep='\t')
    newdf = pd.read_csv('mydf.csv', sep='\t')
    print(newdf)
    print(newdf.head())
    df = pd.read_csv('mydf.csv', sep='\t', usecols=['s', 'x'])
    print(df)

    print("Excise:")
    df = pd.read_csv('files/taxi.csv', usecols=['passenger_count', 'trip_distance', 'total_amount'])
    print(df.head())
    print(df['total_amount'].head())
    print(df['total_amount'].mean())
    print(df[df['trip_distance'] == 0]['total_amount'].count())
    print(df[df['total_amount'] <= 0]['total_amount'].count())
    print(df[df['passenger_count'] == 3]['total_amount'].mean())

    df = pd.read_csv('/etc/passwd', sep=':', comment='#', header=None, names=list('abcdef'))
    print(df.head())

    df = pd.read_csv('files/airports/airlines.dat', sep=',', header=None,
                    usecols=[1, 3, 4, 6], names=['flight', 'code2', 'code3', 'country'])
    print(df.head())
    print(df[df['flight'].str.contains('z')].shape)
    print(df[df['country'] == 'China'].shape)
    print(df['country'].value_counts().head(6))

    print("Replace country with code")
    df['country'] = df['country'].astype('category')
    print(df.head())

def dataframe_corr():
    df = pd.read_csv('files/taxi.csv')
    print(df.head())
    print(df.corr())

    df = pd.read_csv('files/taxi.csv',
                    usecols=['passenger_count',
                             'trip_distance',
                             'total_amount',
                             'tip_amount',
                             'tpep_pickup_datetime',
                             'tpep_dropoff_datetime'],
                    parse_dates=['tpep_pickup_datetime',
                                 'tpep_dropoff_datetime'])
    print(df.head())
    print(df.info())
    df['trip_time'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    print(df.head())
    print(df[df['tpep_pickup_datetime'] > '2015-06-04'].head())

    print("excise:")
    print(df[df['trip_time'] < '00:05:00'].shape)
    print(df[df['trip_time'] > '01:00:00'].shape)
    print(df[df['tpep_pickup_datetime'] < '2015-06-01 12:00:00']['trip_distance'].mean())

    #df.plot.scatter('trip_distance', 'total_amount')
    #plt.show()

    corr = df.corr()['total_amount']
    print(corr[corr>0.7])

def dataframe_datetime():
    df = pd.read_csv('files/taxi.csv',
                    usecols=['passenger_count',
                             'trip_distance',
                             'total_amount',
                             'tip_amount',
                             'tpep_pickup_datetime',
                             'tpep_dropoff_datetime'],
                    parse_dates=['tpep_pickup_datetime',
                                 'tpep_dropoff_datetime'])
    print(df.head())
    df['trip_time'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    print(df.head())
    print(df['tpep_pickup_datetime'].dt.hour.value_counts())
    print(df['tpep_pickup_datetime'].dt.dayofweek.value_counts())

    print("Excise:")
    print("How many taxi rides were there per hour of day")
    print(df['tpep_pickup_datetime'].dt.hour.value_counts())
    print("Were there more taxi rides before 12 noon or after")
    print((df['tpep_pickup_datetime'].dt.hour < 12).value_counts())

    df['trip_time_seconds'] = df['trip_time'].dt.seconds
    # df.plot.scatter(x='total_amount', y='trip_time_seconds')
    # plt.show()

    print("datetime Seris")
    df.set_index('tpep_pickup_datetime', inplace=True)
    print(df.head())
    print(df['2015-06-02 11:20:00':'2015-06-02 11:40:00'].head())

    print("Calculate each day")
    print(df.resample('1D').count())
    print(df.resample('2D').count())
    print(df.resample('1H').count())
    print(df['total_amount'].resample('4H').count())

if __name__ == '__main__':
    #series_basic()
    #series_op()
    #series_adv()
    #pandas_plot()
    #matplotlib_plot()
    #seeborn_plot()

    #dataframe_basic()
    #dataframe_op()
    #dataframe_buildin()
    #dataframe_file()
    #dataframe_corr()
    dataframe_datetime()

    