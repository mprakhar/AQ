
"""
"""
File       grangertest.py

author  Dr. Ernesto P. Adorio
            UPDEPP, UP CLARK
            ernesto.adorio@gmail.com
            
version 0.0.1  06.06.2010
version 0.0.2  06.09.2010
"""

from matlib import *
import lm
from scipy import stats


def grangertest(Y,X,  maxlag):
        """
        Performs a Granger causality test on variables (vectors) Y and X.
        The null hypothesis is: Does X causes Y?
        Returned value: pvalue, F, df1, df2
        """
        # Create linear model involving Y lags only.
        n = len(Y)
        if n != len(X):
           raise ValueError,  "grangertest: incompatible Y,X vectors"
        M = [ [0] * maxlag for i in range(n-maxlag)]
        for i in range(maxlag,  n):
            for j in range(1,  maxlag+1):
                M[i-maxlag][j-1] =  Y[i-j] 
        
        fit = lm.ols(M, Y[maxlag:])
        RSSr = fit.RSS
       
       # Create linear model including X lags.
        for i in range(maxlag,  n):
            xlagged = [X[i-j] for j in range(1,  maxlag+1)]
            M[i-maxlag].extend(xlagged)
        fit = lm.ols(M, Y[maxlag:])
        RSSu = fit.RSS
        df1 = maxlag
        df2 = n - 2 * maxlag - 1
        F = ((RSSr - RSSu)/df1) /(RSSu/df2)
        pvalue = 1.0 - stats.f.cdf(F,  df1,  df2)
        return pvalue, F, df1,  df2,   RSSr,  RSSu
    
def Test():
   D = open("chick-egg.dat").read().split("\n")[1:]
   years,  chicks ,  eggs = [], [],  []
   for line in D:
       splitted = line.split()
       if len(splitted) == 3:
          year,  chick,  egg = splitted
          years.append(float(year))
          chicks.append(float(chick))
          eggs.append(float(egg))
          
   lag = 4
   print "Ho: do chicks cause eggs?" 
   pvalue,  F,  df1,  df2,  RSSr,  RSSu =grangertest(eggs,  chicks,  maxlag = lag)
   print "pvalue, F, df1, df2, RSSr, RSSu=",  pvalue ,  F, df1,  df2,  RSSr,  RSSu
   
   print "Ho: do eggs cause chicks?" 
   pvalue,  F,  df1,  df2,  RSSr,  RSSu =grangertest(chicks, eggs,  maxlag = lag)
   print "pvalue, F, df1, df2, RSSr, RSSu=",  pvalue ,  F, df1,  df2,  RSSr,  RSSu
    
if __name__ == "__main__":        
        Test()
