from sklearn.ensemble import IsolationForest

class CustomIsolationForest:
    def __init__(self, n_estimators=10, contamination=0.002, max_samples = 'auto', max_features = 0.8, random_state=42):
        
        print('The following params are used: n_estimators={}, contamination={}, max_samples={}, max_features = {}, random_state={}'.format(n_estimators,contamination,max_samples,max_features,random_state))

        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, max_features=max_features, random_state=random_state)
        
    def fit(self, X):
        self.model.fit(X)
    
    def predict(self, X):
        y_hat = self.model.predict(X)
        y_hat[y_hat == 1] = 0   # Valid transactions are labelled as 0.
        y_hat[y_hat == -1] = 1  # Fraudulent transactions are labelled as 1.
        return y_hat
    
    def decision_function(self, X):
        return self.model.decision_function(X)
    
    def get_params(self):
        return self.model.get_params()

    def add_results(self,df,y_hat,scores):
        df['Yhat'] = y_hat
        df['scores'] = scores
        return df

