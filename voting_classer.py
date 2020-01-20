from sklearn.ensemble import VotingClassifier
def Voting_Class(m1, m2, x_train, x_test, y_train, y_test)
  m1 = LogisticRegression(random_state=1)
  m2 = tree.DecisionTreeClassifier(random_state=1)
  model = VotingClassifier(estimators=[('lr',model), ('dt', m2)], voting = 'hard')
  model.fit(x_train, y_train)
  model.score(x_test, y_test)
  return model
