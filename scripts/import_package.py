import bank_deposit_classifier.sample as s
import pandas as pd
# s.print_hi()
def data():
    data =pd.DataFrame({ 'y': [0,0,0,0,0,0,0,1,1,1],
    'x1':[1,2,1,2,1,2,1,5,6,5]
    })

    return data
x =data()
s.upsample_minority_class(x,'y',0.5)    
