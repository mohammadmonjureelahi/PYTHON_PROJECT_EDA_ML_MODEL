def modelcomparison(data,targetcolumn):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    pd.options.display.float_format = "{:,.2f}".format
    X = data.drop(targetcolumn,axis=1)
    y = data[targetcolumn]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions_lr = logmodel.predict(X_test)
    dtreemodel = DecisionTreeClassifier()
    dtreemodel.fit(X_train,y_train)
    predictions_dt = dtreemodel.predict(X_test)
    randforestmodel = RandomForestClassifier(n_estimators=600)
    randforestmodel.fit(X_train,y_train)
    predictions_rf = randforestmodel.predict(X_test)
    lr_accuracy = accuracy_score(y_test,predictions_lr)
    dt_accuracy = accuracy_score(y_test,predictions_dt)
    rf_accuracy = accuracy_score(y_test,predictions_rf)
    lr_f1_score = f1_score(y_test,predictions_lr, average=None)
    dt_f1_score = f1_score(y_test,predictions_dt, average=None)
    rf_f1_score = f1_score(y_test,predictions_rf, average=None)
    lr_recall = recall_score(y_test,predictions_lr, average=None)
    dt_recall = recall_score(y_test,predictions_dt, average=None)
    rf_recall = recall_score(y_test,predictions_rf, average=None)
    lr_precision = precision_score(y_test,predictions_lr, average=None)
    dt_precision = precision_score(y_test,predictions_dt, average=None)
    rf_precision = precision_score(y_test,predictions_rf, average=None)
    accuracy_table = {'Accuracy':[lr_accuracy, dt_accuracy, rf_accuracy]}
    accuracy_df = pd.DataFrame(accuracy_table, index =['Logistic Regression', 'Decision Tree', 'Random Forest'])

    precision_recall_f1score_table = {'F1_Score':[lr_f1_score[0], dt_f1_score[0],rf_f1_score[0],lr_f1_score[1],  dt_f1_score[1],
                                                  rf_f1_score[1]],
                                      'Recall':[lr_recall[0],dt_recall[0],rf_recall[0], lr_recall[1], dt_recall[1],  
                                                rf_recall[1]],
                                      'Precision':[lr_precision[0],dt_precision[0],rf_precision[0],lr_precision[1], 
                                                 dt_precision[1],rf_precision[1]]}
    precision_recall_f1score_df = pd.DataFrame(precision_recall_f1score_table, index =['Logistic Regression 0',
                                                                                       'Decision Tree 0','Random Forest 0',
                                                                                       'Logistic Regression 1', 
                                                                                        'Decision Tree 1',
                                                                                        'Random Forest 1'])
    
    print("\nConfusion Matrix for Logistic Regression:\n")
    print(confusion_matrix(y_test,predictions_lr))
    print("\nConfusion Matrix for Decision Tree:\n")
    print(confusion_matrix(y_test,predictions_dt))
    print("\nConfusion Matrix for Random Forest:\n")
    print(confusion_matrix(y_test,predictions_rf))
    print("\nAccuracy for Logistic Regression, Decision Tree and Random Forest :\n")
    print(accuracy_df)
    accuracy_df.plot(kind="bar", color=['coral'],figsize=(12,8), fontsize=15)
    plt.title("Accuracy for Logistic Regression, Decision Tree and Random Forest Models", fontsize=20)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Accuracy of the Model")
    print("\nF1_Score, Recall and Precision for Logistic Regression, Decision Tree and Random Forest :\n\n")
    print(precision_recall_f1score_df)
    print("\n\n")
    plt.figure(figsize=(11,7))
    precision_recall_f1score_df.transpose().plot(kind="bar",figsize=(12,8), fontsize=15)
    plt.title("F1_Score, Recall and Precision for Logistic Regression, Decision Tree and Random Forest Models", fontsize=17)
    plt.xticks(rotation=0, horizontalalignment="center")
    
    


def confusionmatrix(data,targetcolumn):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    pd.options.display.float_format = "{:,.2f}".format
    X = data.drop(targetcolumn,axis=1)
    y = data[targetcolumn]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions_lr = logmodel.predict(X_test)
    dtreemodel = DecisionTreeClassifier()
    dtreemodel.fit(X_train,y_train)
    predictions_dt = dtreemodel.predict(X_test)
    randforestmodel = RandomForestClassifier(n_estimators=600)
    randforestmodel.fit(X_train,y_train)
    predictions_rf = randforestmodel.predict(X_test)

    print("\nConfusion Matrix for Logistic Regression:\n")
    print(confusion_matrix(y_test,predictions_lr))
    print("\nConfusion Matrix for Decision Tree:\n")
    print(confusion_matrix(y_test,predictions_dt))
    print("\nConfusion Matrix for Random Forest:\n")
    print(confusion_matrix(y_test,predictions_rf))
    


def classificationreport(data,targetcolumn):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    pd.options.display.float_format = "{:,.2f}".format
    X = data.drop(targetcolumn,axis=1)
    y = data[targetcolumn]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions_lr = logmodel.predict(X_test)
    dtreemodel = DecisionTreeClassifier()
    dtreemodel.fit(X_train,y_train)
    predictions_dt = dtreemodel.predict(X_test)
    randforestmodel = RandomForestClassifier(n_estimators=600)
    randforestmodel.fit(X_train,y_train)
    predictions_rf = randforestmodel.predict(X_test)
    lr_accuracy = accuracy_score(y_test,predictions_lr)
    dt_accuracy = accuracy_score(y_test,predictions_dt)
    rf_accuracy = accuracy_score(y_test,predictions_rf)
    lr_f1_score = f1_score(y_test,predictions_lr, average=None)
    dt_f1_score = f1_score(y_test,predictions_dt, average=None)
    rf_f1_score = f1_score(y_test,predictions_rf, average=None)
    lr_recall = recall_score(y_test,predictions_lr, average=None)
    dt_recall = recall_score(y_test,predictions_dt, average=None)
    rf_recall = recall_score(y_test,predictions_rf, average=None)
    lr_precision = precision_score(y_test,predictions_lr, average=None)
    dt_precision = precision_score(y_test,predictions_dt, average=None)
    rf_precision = precision_score(y_test,predictions_rf, average=None)
    accuracy_table = {'Accuracy':[lr_accuracy, dt_accuracy, rf_accuracy]}
    accuracy_df = pd.DataFrame(accuracy_table, index =['Logistic Regression', 'Decision Tree', 'Random Forest'])
    
    print("Classification Report for Logistic Regression:\n")
    print(classification_report(y_test,predictions_lr))
    print("Classification Report for Decision Tree:\n")
    print(classification_report(y_test,predictions_dt))
    print("Classification Report for Random Forest:\n")
    print(classification_report(y_test,predictions_rf))



def detailreport(data,targetcolumn):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    pd.options.display.float_format = "{:,.2f}".format
    X = data.drop(targetcolumn,axis=1)
    y = data[targetcolumn]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions_lr = logmodel.predict(X_test)
    dtreemodel = DecisionTreeClassifier()
    dtreemodel.fit(X_train,y_train)
    predictions_dt = dtreemodel.predict(X_test)
    randforestmodel = RandomForestClassifier(n_estimators=600)
    randforestmodel.fit(X_train,y_train)
    predictions_rf = randforestmodel.predict(X_test)
    lrcm = confusion_matrix(y_test,predictions_lr)
    dtcm = confusion_matrix(y_test,predictions_dt)
    rfcm = confusion_matrix(y_test,predictions_rf)
    lr_accuracy = accuracy_score(y_test,predictions_lr)
    dt_accuracy = accuracy_score(y_test,predictions_dt)
    rf_accuracy = accuracy_score(y_test,predictions_rf)
    lr_f1_score = f1_score(y_test,predictions_lr, average=None)
    dt_f1_score = f1_score(y_test,predictions_dt, average=None)
    rf_f1_score = f1_score(y_test,predictions_rf, average=None)
    lr_recall = recall_score(y_test,predictions_lr, average=None)
    dt_recall = recall_score(y_test,predictions_dt, average=None)
    rf_recall = recall_score(y_test,predictions_rf, average=None)
    lr_precision = precision_score(y_test,predictions_lr, average=None)
    dt_precision = precision_score(y_test,predictions_dt, average=None)
    rf_precision = precision_score(y_test,predictions_rf, average=None)
    accuracy_table = {'Accuracy':[lr_accuracy, dt_accuracy, rf_accuracy]}
    accuracy_df = pd.DataFrame(accuracy_table, index =['Logistic Regression', 'Decision Tree', 'Random Forest'])
    precision_recall_f1score_table = {'F1_Score':[lr_f1_score[0], dt_f1_score[0],rf_f1_score[0],lr_f1_score[1],  dt_f1_score[1],
                                                  rf_f1_score[1]],
                                      'Recall':[lr_recall[0],dt_recall[0],rf_recall[0], lr_recall[1], dt_recall[1],  
                                                rf_recall[1]],
                                      'Precision':[lr_precision[0],dt_precision[0],rf_precision[0],lr_precision[1], 
                                                 dt_precision[1],rf_precision[1]]}
    precision_recall_f1score_df = pd.DataFrame(precision_recall_f1score_table, index =['Logistic Regression 0',
                                                                                       'Decision Tree 0','Random Forest 0',
                                                                                       'Logistic Regression 1', 
                                                                                        'Decision Tree 1',
                                                                                        'Random Forest 1'])
    
    print("Classification Report for Logistic Regression:\n")
    print(classification_report(y_test,predictions_lr))
    print("Classification Report for Decision Tree:\n")
    print(classification_report(y_test,predictions_dt))
    print("Classification Report for Random Forest:\n")
    print(classification_report(y_test,predictions_rf))
    print("\nConfusion Matrix for Logistic Regression:\n")
    print(confusion_matrix(y_test,predictions_lr))
    print("\nConfusion Matrix for Decision Tree:\n")
    print(confusion_matrix(y_test,predictions_dt))
    print("\nConfusion Matrix for Random Forest:\n")
    print(confusion_matrix(y_test,predictions_rf))
    print("\nAccuracy for Logistic Regression, Decision Tree and Random Forest :\n")
    print(accuracy_df)
    accuracy_df.plot(kind="bar", color=['coral'],figsize=(12,8), fontsize=15)
    plt.title("Accuracy for Logistic Regression, Decision Tree and Random Forest Models", fontsize=20)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Accuracy of the Model")
    print("\nF1_Score, Recall and Precision for Logistic Regression, Decision Tree and Random Forest :\n\n")
    print(precision_recall_f1score_df)
    print("\n\n")
    plt.figure(figsize=(11,7))
    precision_recall_f1score_df.transpose().plot(kind="bar",figsize=(12,8), fontsize=15)
    plt.title("F1_Score, Recall and Precision for Logistic Regression, Decision Tree and Random Forest Models", fontsize=17)
    plt.xticks(rotation=0, horizontalalignment="center")        



def modelcomparisonplot(data,targetcolumn):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    pd.options.display.float_format = "{:,.2f}".format
    X = data.drop(targetcolumn,axis=1)
    y = data[targetcolumn]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions_lr = logmodel.predict(X_test)
    dtreemodel = DecisionTreeClassifier()
    dtreemodel.fit(X_train,y_train)
    predictions_dt = dtreemodel.predict(X_test)
    randforestmodel = RandomForestClassifier(n_estimators=600)
    randforestmodel.fit(X_train,y_train)
    predictions_rf = randforestmodel.predict(X_test)
    lrcm = confusion_matrix(y_test,predictions_lr)
    dtcm = confusion_matrix(y_test,predictions_dt)
    rfcm = confusion_matrix(y_test,predictions_rf)
    lr_accuracy = accuracy_score(y_test,predictions_lr)
    dt_accuracy = accuracy_score(y_test,predictions_dt)
    rf_accuracy = accuracy_score(y_test,predictions_rf)
    lr_f1_score = f1_score(y_test,predictions_lr, average=None)
    dt_f1_score = f1_score(y_test,predictions_dt, average=None)
    rf_f1_score = f1_score(y_test,predictions_rf, average=None)
    lr_recall = recall_score(y_test,predictions_lr, average=None)
    dt_recall = recall_score(y_test,predictions_dt, average=None)
    rf_recall = recall_score(y_test,predictions_rf, average=None)
    lr_precision = precision_score(y_test,predictions_lr, average=None)
    dt_precision = precision_score(y_test,predictions_dt, average=None)
    rf_precision = precision_score(y_test,predictions_rf, average=None)
    accuracy_table = {'Accuracy':[lr_accuracy, dt_accuracy, rf_accuracy]}
    accuracy_df = pd.DataFrame(accuracy_table, index =['Logistic Regression', 'Decision Tree', 'Random Forest'])

    precision_recall_f1score_table = {'F1_Score':[lr_f1_score[0], dt_f1_score[0],rf_f1_score[0],lr_f1_score[1],  dt_f1_score[1],
                                                  rf_f1_score[1]],
                                      'Recall':[lr_recall[0],dt_recall[0],rf_recall[0], lr_recall[1], dt_recall[1],  
                                                rf_recall[1]],
                                      'Precision':[lr_precision[0],dt_precision[0],rf_precision[0],lr_precision[1], 
                                                 dt_precision[1],rf_precision[1]]}
    precision_recall_f1score_df = pd.DataFrame(precision_recall_f1score_table, index =['Logistic Regression 0',
                                                                                       'Decision Tree 0','Random Forest 0',
                                                                                       'Logistic Regression 1', 
                                                                                        'Decision Tree 1',
                                                                                        'Random Forest 1'])
    
    print(accuracy_df)
    accuracy_df.plot(kind="bar", color=['coral'],figsize=(12,8), fontsize=15)
    plt.title("Accuracy for Logistic Regression, Decision Tree and Random Forest Models", fontsize=20)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Accuracy of the Model")
    print("\nF1_Score, Recall and Precision for Logistic Regression, Decision Tree and Random Forest :\n\n")
    print(precision_recall_f1score_df)
    print("\n\n")
    plt.figure(figsize=(11,7))
    precision_recall_f1score_df.transpose().plot(kind="bar",figsize=(12,8), fontsize=15)
    plt.title("F1_Score, Recall and Precision for Logistic Regression, Decision Tree and Random Forest Models", fontsize=17)
    plt.xticks(rotation=0, horizontalalignment="center")