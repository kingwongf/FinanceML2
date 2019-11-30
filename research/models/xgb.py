import pandas as pd
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from tools.featGen import get_norm_side, moskowitz_func, tanh_func, mrm_c
from sklearn.metrics import classification_report

# from imblearn import

#from tools import visual_tools
import pickle
from scipy.stats.mstats import zscore, winsorize

class xgb(object):
	def __init__(self, Xy_loc, test_date_split, target, side=True):
		self.Xy = pd.read_pickle(Xy_loc)
		self.test_date_split = test_date_split
		self.side = side
		self.target = target


	def set_target(self, col):
		if self.side:
			# self.Xy['target'] = get_norm_side(self.Xy[col], (self.Xy["emaret1m"], self.Xy["retvol1m"], 1.645))
			# self.Xy['target'] = np.sign(self.Xy[col])
			self.Xy['target'] = self.Xy['target'].astype('category')
		# self.Xy['target'] = to_categorical(get_norm_side(self.Xy[col], (self.Xy["emaret"], self.Xy["retvol1m"], 1.645)).astype('category'),3)
		# self.Xy = self.Xy[self.Xy["target"].notnull()]
		else:
			# self.Xy['target'] = self.Xy[col]
			# self.Xy['target'] = moskowitz_func(self.Xy[col])
			# self.Xy['target'] = tanh_func(self.Xy[col])
			self.Xy['target'] = mrm_c(self.Xy[col])

		self.Xy = self.Xy.drop([col], axis=1)

	def train_test_split(self):
		return self.Xy[:self.test_date_split], self.Xy[self.test_date_split:]

	@staticmethod
	def Xy_split(df, label):
		df1 = df.copy()
		target = df1.pop(label)
		return df1, target

	def gen_feature(self):
		categorical_features = self.Xy.columns[(self.Xy.dtypes.values != np.dtype('float64')) & (self.Xy.columns != self.target)].tolist()
		numeric_features = self.Xy.columns[(self.Xy.dtypes.values == np.dtype('float64')) & (self.Xy.columns != self.target)].tolist()
		return categorical_features, numeric_features

	def make_pipeline(self):
		numeric_transformer = Pipeline(
			steps=[('imputer', SimpleImputer(strategy='median')), ('scalar', StandardScaler())])

		categorical_transfomer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
		                                         ('onehot_sparse', OneHotEncoder(handle_unknown='ignore'))])

		categorical_features, numeric_features = self.gen_feature()
		preprocessor = ColumnTransformer(
			transformers=[
				('num', numeric_transformer, numeric_features),
				('cat', categorical_transfomer, categorical_features)
			]
		)

		clf_params = {'num_class': 3, 'objective': 'multi:softprob'}
		reg_params = {'max_depth': 10, 'objective': 'reg:squarederror'}

		if self.side:
			clf = Pipeline(steps=[('preprocessor', preprocessor),
								  ('classifier', XGBClassifier(**clf_params))])
			return clf
		else:
			reg = Pipeline(steps=[('preprocessor', preprocessor),
								  ('classifier', XGBRegressor(**reg_params))])
			return reg



	def fit_pipeline(self):
		print("Preparing to train...")
		# print(train_X.shape)
		self.set_target("fwdret")
		clf = self.make_pipeline()

		train, test = self.train_test_split()
		# print(train, test)
		train_X, train_y = self.Xy_split(train,self.target)
		test_X, test_y = self.Xy_split(test,self.target)

		clf.fit(train_X, train_y)

		pred_y = clf.predict(test_X)
		# np.savetxt(file, pred_y)



		# print("Test Accuracy", accuracy_score(test_y, pred_y))

		if self.side: print(classification_report(test_y, pred_y))
		else: print(r2_score(test_y, pred_y))


xgb_test = xgb("pre_data/feat_useod_daily_1dfwd.pkl", '2018-01-01', "target", False) ## reg
# xgb_test = xgb("pre_data/feat_useod_daily_3dfwd.pkl", '2018-01-01', "target") ## clf
xgb_test.fit_pipeline()

'''
with np.sign 1mfwd
              precision    recall  f1-score   support

        -1.0       0.45      0.06      0.10     90514
         0.0       0.00      0.00      0.00      1990
         1.0       0.52      0.94      0.67     98597

    accuracy                           0.51    191101
   macro avg       0.32      0.33      0.26    191101
weighted avg       0.48      0.51      0.39    191101

with np.sign 1dfwd
              precision    recall  f1-score   support

        -1.0       0.48      0.12      0.19     89061
         0.0       0.00      0.00      0.00      1985
         1.0       0.53      0.89      0.66    100055

    accuracy                           0.52    191101
   macro avg       0.33      0.34      0.28    191101
weighted avg       0.50      0.52      0.43    191101

with np.sign 3dfwd
              precision    recall  f1-score   support

        -1.0       0.48      0.07      0.12     89302
         0.0       0.00      0.00      0.00      1986
         1.0       0.52      0.93      0.67     99813

    accuracy                           0.52    191101
   macro avg       0.33      0.33      0.26    191101
weighted avg       0.50      0.52      0.41    191101

'''