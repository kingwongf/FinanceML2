import pandas as pd
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from tools.featGen import get_norm_side
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
# from imblearn import

#from tools import visual_tools
import pickle
from scipy.stats.mstats import zscore, winsorize

class xgb(object):
	def __init__(self, Xy_loc, test_date_split, target, side=True, save_model=False):
		self.Xy = pd.read_pickle(Xy_loc)
		self.test_date_split = test_date_split
		self.side = side
		self.target = target
		self.model = None
		self.save_model = save_model
		self.categorical_features = None
		self.numeric_features = None

	def set_target(self, col):
		if self.side:
			self.Xy['target'] = get_norm_side(self.Xy[col], (self.Xy["emaret"], self.Xy["retvol1m"], 1.645))
			self.Xy['target'] = self.Xy['target'].astype('category')
		# self.Xy['target'] = to_categorical(get_norm_side(self.Xy[col], (self.Xy["emaret"], self.Xy["retvol1m"], 1.645)).astype('category'),3)
		# self.Xy = self.Xy[self.Xy["target"].notnull()]
		else:
			self.Xy['target'] = self.Xy[col]
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
		self.categorical_features, self.numeric_features = categorical_features, numeric_features
		return categorical_features, numeric_features

	def make_pipeline(self):
		numeric_transformer = Pipeline(
			steps=[('imputer', SimpleImputer(strategy='median')),
				   ('scalar', StandardScaler())])

		categorical_transfomer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
										   ('onehot_sparse', OneHotEncoder(handle_unknown='ignore'))])

		categorical_features, numeric_features = self.gen_feature()
		preprocessor = ColumnTransformer(
			transformers=[
				('num', numeric_transformer, numeric_features),
				('cat', categorical_transfomer, categorical_features)
			]
		)

		clf_params = {'num_class': 3, 'objective': 'multi:softprob'}

		## TODO customise under_sampler
		under_sampler = RandomUnderSampler(random_state=42)
		# under_sampler = NearMiss(random_state=42, version=2)
		# under_sampler = TomekLinks()

		if self.side:
			clf = Pipeline(steps=[('preprocessor', preprocessor),
								  ('under_sampler', under_sampler),
								  ('classifier', XGBClassifier(params=clf_params))])
			self.model = clf
			return clf
		else:
			reg = Pipeline(steps=[('preprocessor', preprocessor),
								  ('under_sampler', under_sampler),
								  ('classifier', XGBRegressor())])
			self.model = reg
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


		print("Test Accuracy", accuracy_score(test_y, pred_y))

		if self.side: print(classification_report(test_y, pred_y))
		else: print(r2_score(test_y, pred_y))
	def imp_feat(self,n):
		onehot_columns = self.model.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
			'onehot_sparse'].get_feature_names(input_features=self.categorical_features)

		feature_importance = pd.Series(data=self.model.named_steps['classifier'].feature_importances_,
									   index=np.append(self.numeric_features, onehot_columns))

		return feature_importance.sort_values(ascending=False)[feature_importance != 0][0:n]


# xgb_reg_test = xgb("pre_data/feat_useod_daily_1mfwd.pkl", '2018-01-01', "target", False)
xgb_clf_test = xgb("pre_data/feat_useod_daily_1mfwd.pkl", '2018-01-01', "target", True)
xgb_clf_test.fit_pipeline()

'''
Random Under-sampler

Test Accuracy 0.5130533068900739
              precision    recall  f1-score   support

        -1.0       0.14      0.50      0.22     14348
         0.0       0.93      0.52      0.67    166054
         1.0       0.10      0.41      0.16     10699

    accuracy                           0.51    191101
   macro avg       0.39      0.48      0.35    191101
weighted avg       0.82      0.51      0.60    191101


'''
