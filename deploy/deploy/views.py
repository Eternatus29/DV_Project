from django.shortcuts import render,HttpResponse

# Create your views here.
def index(request):
    size=request.POST.get('size')
    cyl=request.POST.get('cyl')
    fuel=request.POST.get('fuel')



        
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    if (size!=None):
            size=float(size)
            cyl=float(cyl)
            fuel=float(fuel)
    # Load the csv file
    df = pd.read_csv("templates\FuelConsumption.csv")
    df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


    # Select independent and dependent variable
    
    X_train = np.array(df.iloc[:, 0:3])
    y_train = np.array(df.iloc[:, 3:])
    
    regr=LinearRegression()
    regr.fit(X_train,y_train)
    # Feature scaling

    
    prediction=[0]
    if (size!=None):
            
          prediction[0]=float(regr.predict(np.array([[size,cyl,fuel]]))[0])

    prediction=float(prediction[0])
    print(prediction)
    return render(request,'index.html',{'prediction':prediction})
