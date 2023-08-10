import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import cross_val_score, KFold, RepeatedStratifiedKFold
import pickle

otu_table = pd.read_csv('species.csv', index_col=0)
labels = pd.read_csv('metadata.csv', index_col=0) 
labels1= LabelEncoder().fit_transform(labels['Group'])

X = otu_table.iloc[:, 1:].values
y = labels1

original_feature_names = otu_table.columns[1:].tolist()

# f_classif
f_selector = SelectKBest(score_func=f_classif)
X_f_selected = f_selector.fit_transform(X, labels1)
f_selected_feature_indices = f_selector.get_support(indices=True)
f_selected_feature_names = [original_feature_names[i] for i in f_selected_feature_indices]

# mutual_info_classif
mutual_selector = SelectKBest(score_func=mutual_info_classif)
X_mutual_selected = mutual_selector.fit_transform(X, labels1)
mutual_selected_feature_indices = mutual_selector.get_support(indices=True)
mutual_selected_feature_names = [original_feature_names[i] for i in mutual_selected_feature_indices]

# RF
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
# GB
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# XGB
xgb_model = XGBClassifier(n_jobs=1, random_state=0)
# LGB
lgb_model = LGBMClassifier(random_state=0)
# L1
lr_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=0)

#Extracting subsets of data
all_importances = []
methods = ['f_classif', 'Mutual Information', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'L1 Regularization']
selected_feature_names_dict = {}

for method, selector in zip(['f_classif', 'Mutual Information'], [f_selector, mutual_selector]):
    method_selected_feature_indices = np.where(selector.scores_ > 0.001)[0]
    method_selected_feature_names = [original_feature_names[i] for i in method_selected_feature_indices]
    selected_feature_names_dict[method] = pd.Series(method_selected_feature_names)

for model, name in zip([rf_model, gb_model, xgb_model, lgb_model, lr_model], ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'L1 Regularization']):
    model.fit(X, labels1) 
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_[0]
    all_importances.append(importances)

    method_selected_feature_indices = np.where(importances > 0.001)[0]
    method_selected_feature_names = [original_feature_names[i] for i in method_selected_feature_indices]
    selected_feature_names_dict[name] = pd.Series(method_selected_feature_names)

result_dfs = []
for method, features in selected_feature_names_dict.items():
    selected_features = otu_table[features.dropna().values]
    result_df = pd.concat([otu_table.iloc[:, 0], selected_features], axis=1)
    result_dfs.append(result_df)

# TO_CSV
for method, result_df in zip(methods, result_dfs):
    result_df.to_csv(f"feature_selection_{method.replace(' ', '_')}_results.csv", index=False)



# Classification Performance Evaluation
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from sklearn.model_selection import cross_val_score, RepeatedKFold, RepeatedStratifiedKFold


methods = ['Oringin','Univariate ANOVA', 'Mutual Information', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'L1 Regularization']

data_frames = {}
for method in methods:
    file_name = f"feature_selection_{method.replace(' ', '_')}_results.csv"
    data_frames[method] = pd.read_csv(file_name)

results = {}
for df, name in zip(dataframes, df_names):
    labs = []
    for i in range(len(df)):
        labs.append(df.iloc[i, 1])
    labs = pd.DataFrame(labs)

    lab = labs
    enc = LabelEncoder()
    labe = enc.fit_transform(lab.values)

    X = np.array(df.iloc[:, 2:])
    y = np.array(labs)

    models = [KNeighborsClassifier(), SVC(probability=True, random_state=0), DecisionTreeClassifier(random_state=0),
              RandomForestClassifier(random_state=0), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), XGBClassifier(use_label_encoder=False, n_jobs=1, random_state=0), 
              XGBRFClassifier(use_label_encoder=False, n_jobs=1, random_state=0), LGBMClassifier(random_state=0)]
    names = ['KNN', 'SVM', 'DT', 'RF', 'GB', 'XGB', 'XGBRF', 'LGB' ]

    cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

    aucs = []
    for i, n in zip(names, models):
        print(f"{name} - {i}")
        auc = cross_val_score(n, df.iloc[:, 2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        aucs.append(auc)

    results[name] = pd.Series(aucs, index=names)

combined_results = pd.DataFrame(results)