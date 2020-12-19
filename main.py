
class Preprocesser:
    
    def __init__(self,num_columns,cat_columns):
        self.num_fill_value       = {}
        self.cat_fill_value       = {}
        self.numerical_features   = num_columns
        self.categorical_features = cat_columns
        
    def dollar_to_numeric(self,data):
        
        """casting string dollar amounts columns to numerical"""
        
        for col in ['INCOME','OLDCLAIM','BLUEBOOK','HOME_VAL']:
            data[col] = data[col].str.replace('$','').str.replace(',','.')
            data[col] = pd.to_numeric(data[col],downcast='float')
            
    def correct_typos(self,data):
        """ correcting misspelled values"""
        data['EDUCATION']= data['EDUCATION'].replace(to_replace = {'<High School':'z_High School'})
        
    def fillna(self,data,fit=True):
        """ filling missing values using mean value and most frequent value for numerical 
        and categorical columns respectively. Filling values are stored when fit=True"""
        
        
        if fit:
            for col in self.numerical_features:
                self.num_fill_value[col] = np.mean(data[col])
            for col in self.categorical_features:
                self.cat_fill_value[col] = data[col].value_counts().index[0]
        else:
            
            for col in self.numerical_features:
                data[col].fillna(self.num_fill_value[col],inplace=True)

            for col in self.categorical_features: 
                data[col].fillna(self.cat_fill_value[col],inplace=True)
        
    def fit(self,X,y=None):
        """ fitting on training dataset """
        X_ = X.copy()
        self.dollar_to_numeric(X_)
        self.correct_typos(X_)
        self.fillna(X_,fit=True)
        return self
        
    def transform(self,data):
        """ transforming dataset"""
        data = data.copy()
        self.dollar_to_numeric(data)
        self.correct_typos(data)
        self.fillna(data,fit=False)
        return data

class CategoricalEncoder:
    """ encode categorical columns of the training data and store the used strategy"""
    def __init__(self,cat_columns,strategy='onehot'):
        self.fitting_strategy     = strategy
        self.categorical_features = cat_columns
        self.X = None
        
    def fit(self,X,y=None):
        
        X = X.copy()
        if self.fitting_strategy  == "ordinal":
            for col in self.categorical_features:
                X[col] =  X[col].astype('category').cat.codes
            self.X = X
            
        return self
    
                
    def transform(self,X):
        """ encoding test data using the same fitting strategy applied to training data"""
        
        self.fit(X)
        return self.X

if __name__ == "__main__":
    
    import sys
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    data_path = sys.argv[1]

    #reading data
    train_auto = pd.read_csv(data_path + 'train_auto.csv',sep=",",index_col="INDEX")
    test_auto = pd.read_csv(data_path + 'test_auto.csv',sep=",")
    test_auto[['TARGET_FLAG','TARGET_AMT']] = pd.read_csv(data_path + 'test_auto.csv',sep=r",+",usecols=[1,2],engine='python')
    test_auto = test_auto.set_index('INDEX')
    
    # inferring categorical data
    target_columns = ['TARGET_FLAG']
    features = [col for col in train_auto.columns if col not in target_columns]

    categorical_features = []
    numerical_features = []

    for col in features :
        if np.issubdtype('O', train_auto[col].dtype):
            categorical_features.append(col)  
        else:
            numerical_features.append(col)
    
    # split target and features
    X,  y          = train_auto[features],  train_auto[target_columns]
    X_test, y_test = test_auto[features], test_auto[target_columns]
    
    # testing three different algorithms
    dict_models = {}
    
    dict_models[RandomForestClassifier(class_weight='balanced')] = {'n_estimators': range(50,100)}
    dict_models[KNeighborsClassifier()] = {'n_neighbors': range(5,15), 'weights': ['uniform', 'distance']}
    dict_models[SVC(class_weight='balanced')] = {'C': [0.1, 1, 10]}
    # storing cross validation scores in order to choose the best model
    dict_score = {'f1_score':{}}
    for model, grid in dict_models.items():
            pipe = Pipeline([('preprocessor',Preprocesser(num_columns=numerical_features,cat_columns=categorical_features)),
                    ('encoder',CategoricalEncoder(cat_columns=categorical_features,strategy="ordinal")),
                    ('scaler',StandardScaler()),
                    ('pca',PCA(n_components=15)), # 15 components are enough toexplain over 80% of variance in the data
                    ('gsCV',GridSearchCV(model,param_grid=grid,scoring='f1_weighted'))
                    ])
            pipe.fit(X,y['TARGET_FLAG'])
            score_ = round(pipe['gsCV'].best_score_,2)
            dict_score['f1_score'][type(model).__name__] = score_
    
    y_out = pd.DataFrame({'INDEX':X_test.index,'p_target':np.zeros(len(X_test.index))})
    # prediction using the best performing model, we use the last one in the dict
    y_out['p_target'] = pipe.predict(X_test)
    y_out = y_out.set_index('INDEX')
    print(pd.DataFrame(dict_score))
    print(type(model).__name__ + ' f1 s score on test set: ', round(pipe.score(X_test,y_test),2))
    y_out.to_csv(data_path + '/PREDICTED_FLAG.csv')
    
    