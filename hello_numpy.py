import numpy as np



def basic_array():
    print(np.arange(1, 20, 3))
    print(np.array([1, 1, 2 ,3, 5, 8, 11]))
    print(np.random.randint(5))
    print(np.random.rand(10))
    print(np.random.rand(10, 3, 2))
    print(np.ones(6))
    print(np.zeros(20))

def array_ops():
    a = np.arange(10, 60, 10)
    print(a)
    print(a[[True, False, False,True, False]])
    print(a+5)
    print(a/5)

    print(np.ones(10) * 4)
    print(np.zeros(4) + 7.3)

    a = np.ones(10) * 3
    b = np.ones(10) + 4
    print(a+b)

    print("excises:")
    print(np.ones(10) + 2)
    a = np.random.rand(5)
    b = np.random.rand(5)
    print(a)
    print(b)
    print(a+b)
    print(np.random.randint(0, 50, 10))
    print(np.arange(100, 1000, 20)/6)

    a = np.random.randint(0, 1000, 10)
    print(a)
    print(a[:4])
    print(a.mean())
    print(a.std())
    print(a.max())
    print(a[[2,5,7]])

    print("excises:")
    a = np.random.randint(0, 100, 10)
    print(a)
    print(a<40)
    print(a[a<40])
    print(a.mean())

    print(a[a<a.mean()])

    print(a[(a>70) | (a<30)])
    print(a[a%2 == 1])

    print("excises:")
    a = np.random.randint(0, 100, 10)
    print(a)
    print(a.mean())
    print(a.std())
    print(a[a%2 == 0])
    print(a[(a%2 == 1) & (a>a.mean())])
    print(a[(a<a.mean()-a.std()) | (a>a.mean()+a.std())])


def array_dtype():
    a = np.array([1, 1, 2, 3, 4])
    print(a.dtype)
    b = np.array([1, 2, 3, 4.1])
    print(b.dtype)
    a.dtype = np.dtype('float64')
    print(a)

    a = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    print(a)
    print(len(a))
    print(a.size)
    print(a.itemsize)
    a.dtype = np.int32
    print(a)
    print(len(a))
    print(a.size)
    print(a.itemsize)


    a = np.array([10, 20, 30, 40], dtype=np.int64)
    a.dtype = np.float64
    print(a)
    a.dtype = np.int64
    print(a)
    print(a.astype(np.float64))

    print("Excises:")
    a = np.random.randint(0, 1000, 10)
    print(a)
    a = a.astype(np.float64)
    print(a)
    a = a/3
    print(a)
    print(a.mean())

    a = np.random.rand(10) * 1000
    print(a[a.astype(np.int64)%2==0])

    # NOTE: Use smallest dtype that meet the requirements t save cost
    # Ex: use int8 instead of int64

def array_nan():
    a = np.array([90, 95, np.nan, 92, 80])
    print(a)
    print(a.mean())
    print(a[~np.isnan(a)].mean())

    print("excise:")
    a = np.random.rand(20) * 1000
    print(a)
    ft = (a<a.mean()-a.std()) | (a>a.mean()+a.std())
    a[ft] = np.nan
    print(a)
    print(a[~ft].max())
    print(a[~ft].min())
    print(a[~ft].mean())
    print(a[~ft].mean())
    a[np.isnan(a)] = a[~ft].mean()
    print(a)
    print(a.mean())

def array_shape():
    a = np.arange(10, 100, 10)
    print(a)
    print(a.shape)
    a.shape = (3, 3)
    print(a)
    print(a[1, 2])

    a = np.arange(25, 50)
    a.shape = (5, 5)
    print(a)
    print(a[2])
    print(a[2:])
    print(a[:, :3])
    print(a[2:, :3])

    print("excise:")
    a = np.random.randint(0, 1000, 25)
    a = np.random.randint(0, 1000, 25).reshape(5, 5)
    a.shape = (5, 5)
    print(a)
    print(a[1].max())
    print(a[:, 2][a[:, 2]>500].min())
    print(a[[1, 3]].sum())
    print(a[:, 4][a[:, 4]%2 == 0])

def array_sort():
    a = np.random.randint(0, 1000, 25).reshape(5, 5)
    print(a)
    a.sort()
    print(a)
    a = np.random.randint(0, 1000, 25).reshape(5, 5)
    print(a)
    a.sort(axis=0)
    print(a)
    a = np.random.randint(0, 1000, 25).reshape(5, 5)
    print(a)
    a.sort(axis=1)
    print(a)


if __name__ == '__main__':
    print(np.version.full_version)
    basic_array()
    array_ops()
    array_dtype()
    array_nan()
    array_shape()
    array_sort()