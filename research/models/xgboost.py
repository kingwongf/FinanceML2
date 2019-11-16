import pandas as pd
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tools.featGen import get_norm_side

#from tools import visual_tools
import pickle
from scipy.stats.mstats import zscore, winsorize

class xgb(object):
	def __init__(self, Xy_loc, cat, drop_col, test_date_split, batch_size=32, side=True):
		self.Xy = pd.read_pickle(Xy_loc)
		self.cat = cat  ## sould be ['Date', 'target', 'ticker']
		self.drop_cat_leakage_col = drop_col
		self.test_date_split = test_date_split
		self.numeric = [e for e in self.Xy.columns.tolist() if e not in self.drop_cat_leakage_col]
		self.batch_size = batch_size
		self.side = side


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
		return self.Xy[self.test_date_split], self.Xy[self.test_date_split:]

	@staticmethod
	def Xy_split(df, label):
		df1 = df.copy()
		target = df1.pop(label)
		return df, target

	def make_pipeline(self):
		categorical_features = self.Xy.columns[(self.Xy.dtypes.values != np.dtype('float64'))].tolist()
		numeric_features = self.Xy.columns[(self.Xy.dtypes.values == np.dtype('float64'))].tolist()

		numeric_transformer = Pipeline(
			steps=[('imputer', SimpleImputer(strategy='median')), ('scalar', StandardScaler())])

		categorical_transfomer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
		                                         ('onehot_sparse', OneHotEncoder(handle_unknown='ignore'))])

		preprocessor = ColumnTransformer(
			transformers=[
				('num', numeric_transformer, numeric_features),
				('cat', categorical_transfomer, categorical_features)
			]
		)

		params = {'num_class': 3, 'objective': 'multi:softprob'}

		clf = Pipeline(steps=[('preprocessor', preprocessor),
		                      ('classifier', XGBClassifier(params=params))])
		print("Preparing to train...")
		# print(train_X.shape)
		clf.fit(train_X, train_y)


def xgboost_model(file, risk=True, momentum=True, supplychain=True):


	out = out.drop(["datepll", "datepll.1"], axis=1)
	if not risk:
		out = out.drop(['crating', 'orating',
		                'history', 'cond_ind', 'finance', 'moved', 'sales', 'hicdtavg',
		                'pexp_s_n', 'pexp_30', 'pexp_60', 'pexp_90', 'pexp_180', 'bnkrpt',
		                'dbt_ind', 'uccfilng', 'cscore', 'cpct', 'fpct', 'paynorm', 'pubpvt',
		                'pex_sn1', 'bd_ind'], axis=1)

	if not momentum:
		out = out.drop(["mom"], axis=1)

	if not supplychain:
		out = out.drop(['revenue_dependency', 'adj_close_dependency', 'mom_dependency',
		                'vol_dependency', 'MACD_dependency'], axis=1)

	# out = out.drop(['index'], axis=1)
	out['Date'] = pd.to_datetime(out['Date'])
	# out.info()
	# exit(0)
	# out['cat_feat_1'] = out['fwd_return'].astype(str)

	# out = out.replace([np.inf, -np.inf], np.nan)
	# out = out.dropna()
	col_li = out.columns.tolist()

	train_X, test_X = out[out['Date'] <= '2018-12-31'].drop(
		["ReturnClassifier", "excess_return", "fwd_return", "Date", "ticker"], axis=1), out[
		                  out['Date'] > '2018-12-31'].drop(
		["excess_return", "ReturnClassifier", "fwd_return", "Date", "ticker"], axis=1)
	train_y, test_y = out[out['Date'] <= '2018-12-31']['ReturnClassifier'], out[out['Date'] > '2018-12-31'][
		'ReturnClassifier']

	# out[out['Date']>'2018-12-31']["Date"].to_pickle("date.pkl")
	# np.savetxt("ticker.txt", out[out['Date']>'2018-12-31']["ticker"], fmt='%s;')

	categorical_features = out.columns[(out.dtypes.values != np.dtype('float64'))].tolist()
	numeric_features = out.columns[(out.dtypes.values == np.dtype('float64'))].tolist()

	numeric_features.remove('ReturnClassifier')
	numeric_features.remove('fwd_return')
	numeric_features.remove('excess_return')
	categorical_features.remove('Date')
	categorical_features.remove('ticker')

	numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scalar', StandardScaler())])

	categorical_transfomer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
	                                         ('onehot_sparse', OneHotEncoder(handle_unknown='ignore'))])

	preprocessor = ColumnTransformer(
		transformers=[
			('num', numeric_transformer, numeric_features),
			('cat', categorical_transfomer, categorical_features)
		]
	)

	params = {'num_class': 3, 'objective': 'multi:softprob'}

	clf = Pipeline(steps=[('preprocessor', preprocessor),
	                      ('classifier', XGBClassifier(params=params))])
	print("Preparing to train...")
	# print(train_X.shape)
	clf.fit(train_X, train_y)

	# pickle.dumps(clf, open('clf.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	pred_y = clf.predict(test_X)
	# np.savetxt(file, pred_y)

	print(accuracy_score(test_y, pred_y))

	onehot_columns = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
		'onehot_sparse'].get_feature_names(input_features=categorical_features)

	feature_importance = pd.Series(data=clf.named_steps['classifier'].feature_importances_,
	                               index=np.append(numeric_features, onehot_columns))

	feature_importance = feature_importance.sort_values(ascending=False)

	imp_coef = feature_importance.copy()
	imp_coef = imp_coef[imp_coef != 0][0:20]
	imp_coef.to_pickle(file)