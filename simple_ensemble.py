"""Takes 2 models and the training, testing data and returns the final prediction"""
#Should all the data be stored in a series? List? To be easier to apss to various functions?

def simple_ensemble(m1,m2,m3,x_train,ytrain,x_test):

  m1.fit(x_train, y_train)
  m2.fit(x_train, y_train)
  m3.fit(x_train, y_train)

  p1=m1.predict(x_test)
  pt2=m2.predict(x_test)
  p3=m3.predict(x_test)

  final_prediction = np.array([])
  for i in range(0,len(x_test)):
    final_prediction = np.append(final_prediction, mode([p1[i], p2[i], p3[i]]))
    
  return final_prediction
