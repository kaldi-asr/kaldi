参考https://applenob.github.io/hmm.html#%E5%BD%A2%E5%BC%8F%E5%AE%9A%E4%B9%89

###   numpy.random.multinomial (n，pvals，size = None) 
参考https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html
n：int 实验次数;   pvals：长度序列，表示每次的概率p;   size : int or tuple of ints, optional，要输出几遍这样的序列。

1.投一个均匀骰子20次：

    np.random.multinomial(20, [1/6.]*6, size=1)
    array([[4, 1, 7, 5, 2, 1]]) （20次中，输出1、2、3、4、5、6的次数分别是4、1、7、5、2、1）

2.投一个不均匀骰子100次，投到6的概率是2/7：

    np.random.multinomial(100, [1/7.]*5 + [2/7.])
    array([11, 16, 14, 17, 16, 26])

3.
    
    >>> np.random.multinomial(20, [1/6.]*6, size=2)
            array([[3, 4, 3, 3, 4, 3],
                   [2, 4, 3, 4, 0, 7]])
生成两组20次的。

### np.where

    >>> x = np.arange(9.).reshape(3, 3)
    >>> np.where( x > 5 )
    (array([2, 2, 2]), array([0, 1, 2]))
    注释：
    x = array([[ 0.,  1.,  2.],
              [ 3.,  4.,  5.],
              [ 6.,  7.,  8.]])
  而array([2, 2, 2]),  // x轴 
    array([0, 1, 2])   // y轴
  所以输出是坐标 (2,0),(2,1),(2,2)
