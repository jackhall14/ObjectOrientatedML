# Author: Jack Hall
# Last Modified: 23/5/2020
# Things to do: Maybe make the title and xlabels attributes of the classes
# Usuage: python ML.py <optional arugments -DT / -DA / -LR / -PCA / -SVM / -PLOTS> <full path to file + data(.csv)>
# 
# Order of next inclusion:
# discriminant_analysis.py - Need to go through the PDF and see whats going on in relation do the code placement
# logisitic_regression.py
# decision_trees.py
# svm.py
# Just to make plots of variables

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set_style("whitegrid")

def DoPlots(File):
	# Need to develop these plots
	df_obj = DataFrame(File)
	# Convert a column to datatime64?
	df_obj.DF['Post Date'] = pd.to_datetime(df_obj.DF['Post Date'])
	print(df_obj.DF.head())

	x_var = "Post Date"
	x_var = "Sale Item"
	y_var = "Asking Price"
	figureProp_Array = df_obj.setTitleAndCoordLabels("Plot of " + y_var + " against " + x_var, x_var, y_var)

	df_obj.drawBasicScatterPlot(figureProp_Array, x_var, y_var)
	input()

def DoPCA(File):
	# PCA Analysis
	# Try with: irisnf.csv
	df_PCA_obj = PCA(File)
	df_PCA_obj.performPCAAnalysis(3)

	if args.view_DF: df_PCA_obj.viewDF()

	# # Standard plots (works out of the box) - next line seems redundant:
	# figureProp_Array = DataFrame.setTitleAndCoordLabels("Plot of lengths", 'Petal length', 'Sepal length')
	# Series of cool plots:
	# I just chose variables for you but feel to choose your own:
	df_PCA_obj.drawJointKDEPlot(df_PCA_obj.columns[0], df_PCA_obj.columns[1])
	# Cum. var. against number of components
	df_PCA_obj.drawCumaltiveVarPlot()
	# Drawings of the components
	df_PCA_obj.drawPCcomparison(1,2)

def DoDA(File):
	# Discriminant analysis
	# Try with: crabs.csv
	df_DA_obj = DA(File)

	if args.do_da_HELP: df_DA_obj.HelpMode()
	
	# Vars to blah
	discriminant_vars = ["sp","sex"]
	# The variables we will be using to help us to discriminate
	input_vars = ['FL','RW','CL','CW','BD']
	df_DA_obj.performDAAnalysis(discriminant_vars, input_vars)

	if args.view_DF: df_DA_obj.viewDF()
	input()

def main():
	args = get_args()
	File = args.input_file

	if args.do_plots: DoPlots(File)
	if args.do_pca: DoPCA(File)
	if args.do_da: DoDa(File)
	print("Finished.")

def get_args():
	args.add_argument('input_file', help='full path to file /<some file.csv>')
	args.add_argument('-PLOTS','--do_plots',action='store_true', help='For exploratory plots')
	args.add_argument('-PCA','--do_pca',action='store_true', help='Perform Principal component analysis on input csv file. Note, has to be numeric data.')
	args.add_argument('-DA','--do_da',action='store_true', help='Perform discriminant analysis on input csv file. Does this need to be numeric data?')
	args.add_argument('-DA_HELP','--do_da_HELP',action='store_true', help='True in combo with -DA for info on analysis and workflow.')
	args.add_argument('-ViewDF','--view_DF',action='store_true', help='For debugging, desgined to check the DF before plotting.')
	# parser.add_argument('--hue', type=str, help='This option needs variable to seperate colours in sns scatterplots')
	# parser.add_argument('-f','--file',type=str, help='Blanck example')
	return args.parse_args()

if __name__ == '__main__': main()

class DataFrame():
	# Custom DF class
	def __init__(self, filename_path):
		# Full path including filename
		self.filename_path = filename_path
		# Just the filename which is derived from the previous full path + the name
		self.filename = filename_path.split("/"[-1])
		# Create a DF using the file from the full path + name attribute
		self.DF = pd.read_csv(self.filename_path)
		# Also create an attribute for just the path to the file
		array = filename_path.split("/")[1:-1]
		path_to_filename = "/" + '/'.join(array) + "/"
		self.path_to_filename = path_to_filename

	def viewDF(self):
		print(self.DF.describe(include="all"))
		print("\nPress enter to continue")
		input()

	# Create lots of function here for importing a dataframe

	# Plotting functions begin
	# Also include plotting functions here to be called upon by ML classes
	@staticmethod
	def Plot(g):
		plt.show(g)
		print(plt.figure())
		print(plt.axes())

	@staticmethod
	def setTitleAndCoordLabels(title,xlabel,ylabel):
		figureTitle = title
		figurexlabel = xlabel
		figureylabel = ylabel
		figureProp_Array = [figureTitle, figurexlabel, figureylabel]
		return figureProp_Array

	def SetProperties(g, Prop_array):
		g.figure.suptitle(Prop_array[0],fontsize=16)
		plt.xlabel(Prop_array[1],fontsize=14)
		plt.ylabel(Prop_array[2],fontsize=14)

	# Actual plotting functions now
	def drawBasicScatterPlot(self, Prop_array, x_var, y_var):
		# Plot 2 vars, not the prettiest
		# eg of args: "Petal.l", "Sepal.l", figureProp_Array
		g = sns.stripplot(x=x_var,y=y_var,data=self.DF)
		DataFrame.SetProperties(g,Prop_array)
		DataFrame.Plot(g)

	def drawLinearRegression(self,x_var, y_var, Prop_array):
		# Linear regression of two variables
		g = sns.regplot(x=x_var,y=y_var,fit_reg=True,data=self.DF)
		DataFrame.SetProperties(g,Prop_array)
		DataFrame.Plot(g)
		# Note cannot get model parameters usings this sns.regplot

	def drawPariPlot(self):
		# Produces lots of correlation plots between the variables
		g = sns.pairplot(self.DF)
		DataFrame.Plot(g)

	def drawFacetGrid(self, x_var, y_var,comp_var):
		# Plot 2 vars against each other based on another var
		# eg of args: "Petal.l", "Sepal.l", "Variety"
		g = sns.FacetGrid(data=self.DF,col=comp_var) 	#Col = column NOT colour
		g = (g.map(plt.scatter,x_var,y_var).add_legend())
		DataFrame.Plot(g)

	def drawJointPlot(self, x_var,y_var):
		# Scatterplot but with histos on the axis'
		g = sns.jointplot(x_var,y_var,data=self.DF)
		DataFrame.Plot(g)

	def drawJointKDEPlot(self, x_var,y_var):
		# Contour plot but with guassian histos on the axis'
		g = sns.jointplot(x_var,y_var,data=self.DF, kind='kde')
		DataFrame.Plot(g)

from sklearn import metrics
class ConfusionMatrix():
	def init(self):
		print("The Confusion Matrix gives a measure of how well the model performs on the data set.")

	def GetConfusionMatrix(self, ClassVariable, PredictedClassVariable):
		self.confusion_matrix = metrics.confusion_matrix(ClassVariable,PredictedClassVariable)
		return self.confusion_matrix

	def PlotActualVsPredicted(self, ClassVariable, PredictedClassVariable):
		plt.figure(figsize=(6,6))
		sns.heatmap(self.confusion_matrix,annot=True,fmt="d",linewidths=.5,square=True,cmap="Blues_r")
		plt.ylabel("Actual label")
		plt.xlabel("Predicted label")
		plt.show()
		print("Please press enter to continue.")
		input()


######################################
######################################
# Machine Learning Algorithm Classes #
######################################
######################################

# PCA Imports
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
class PCA(DataFrame):
	def init(self):
		pass

	def performPCAAnalysis(self, no_components=2, pca_type = 'correlation'):
		df = self.DF

		self.columns = df.T.index
		self.rownames = df.T.columns
		self.covMatrix = df.cov()
		self.corrMatrix = df.corr()

		print("\n The variance matrix is:")
		print(self.covMatrix)								#variance matrix
		print("\n The correlation matrix is:")
		print(self.corrMatrix)								#correlation matrix

		if pca_type == 'correlation':
			scaler = StandardScaler()
			scaler.fit(df)
			df1 = scaler.transform(df)

			pca = sklearnPCA(n_components=no_components)
			pca.fit(df1)
			self.PCA_eigenvalues = pca.explained_variance_ratio_
			print("\n PCA Eigenvalues:")
			print(self.PCA_eigenvalues) 					#PCA PCA_eigenvalues
			self.PCA_components = pca.components_
			print("\n PCA Components:")
			print(self.PCA_components)						#PCA Components

			self.rotated_pcs = pca.fit_transform(df1)
			print("\n All the results rotated onto PCs:")
			print(self.rotated_pcs)
		else:
			pca = PCA()
			pca.fit(df)										#Optional:PCA(n_components=5)
			print("\n PCA Eigenvalues:")
			print(pca.explained_variance_ratio_)			#PCA Eigenvalues
			print("\n Principal Components:")
			print(pca.components_)

	def drawCumaltiveVarPlot(self):
		#Cumulative variance plot:
		cumexp = np.concatenate([[0],self.PCA_eigenvalues])
		g = sns.distplot(np.cumsum(cumexp))
		Prop_Array = DataFrame.setTitleAndCoordLabels("Title", 'number of components', 'cumulative variance')
		DataFrame.SetProperties(g,Prop_Array)
		DataFrame.Plot(g)

	def drawPCcomparison(self, pc_1, pc_2):
		Prop_Array = DataFrame.setTitleAndCoordLabels("Title", "Principal Component: " + str(pc_1), "Principal Component: " + str(pc_2))
		g = DataFrame.drawBasicScatterPlot(self, Prop_Array, self.rotated_pcs[:,(pc_1-1)], self.rotated_pcs[:,(pc_2-1)])
		DataFrame.SetProperties(g,Prop_Array)
		DataFrame.Plot(g)

# Discriminant Analysis Imports
# from sklearn.preprocessing import StandardScaler			# Already imported from PCA
# from sklearn.decomposition import PCA 					# Already imported from PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
class DA(DataFrame):
	# Discriminant Analysis
	def init(self):
		pass

	def performDAAnalysis(self, DiscriminantVars, ListOfInputVars):
		df = self.DF
		
		# Because of labsheet (5), we begin with a PCA analysis
		X=df[ListOfInputVars].values
		# Transform class variables into vectors
		y1=df[DiscriminantVars[0]].values
		y2=df[DiscriminantVars[1]].values

		# Label Encoder changes values into numerical values
		enc=LabelEncoder()
		lab_enc1=enc.fit(y1)
		y1=lab_enc1.transform(y1)
		lab_enc2=enc.fit(y2)
		y2=lab_enc2.transform(y2)
		# Combine two class variables into one
		y=2*y1-y2+2

		# Figure out a way to turn this to lines 221 to PCA class
		# Works with correlation matrix
		scaler=StandardScaler()
		scaler.fit(X)
		X1=scaler.transform(X)

		pca=sklearnPCA()
		pca.fit(X1)
		p=pca.fit_transform(X1)

		# print("\n PCA Eigenvalues:")
		# print(pca.explained_variance_ratio_)

		# Plot the PCs and colour them by the y (class variable value)
		plt.scatter(x=p[:,0],y=p[:,1],c=y)
		plt.xlabel("PC1")
		plt.ylabel("PC2")
		# plt.show()
		# input()

		plt.scatter(x=p[:,1],y=p[:,2],c=y)
		plt.xlabel("PC2")
		plt.ylabel("PC3")
		# plt.show()
		# input()

		# Beginning of linear discriminant analysis
		lda=LDA(store_covariance=True)
		# Use the previously defined X and y
		lda.fit(X,y)

		print("\n LDA Scalings of eigenvectors W^-1 B")
		print(lda.scalings_)

		print("\n LDA Eigenvalues (sizes of the eigenvalues):")
		print(lda.explained_variance_ratio_)

		print("\n LDA Covariance:")
		print(lda.covariance_)

		# Predicted classes for the original data; rotates matrix X of obsv. onto the discriminant coords
		y_pred=lda.fit(X,y).predict(X)
		# print("\n Y Predictions:")
		# print(y_pred)

		# Test the y predictions:
		cm = ConfusionMatrix()
		cm.GetConfusionMatrix(y,y_pred)
		print("\n Confusion Matrix:")
		print(cm.confusion_matrix)
		# Plot it nicely?
		# cm.PlotActualVsPredicted(y,y_pred)

		# Probs to each point in space along decision boundaries for groups
		y_predprob=lda.fit(X,y).predict_proba(X)
		# print("\n Predicted Probability:")
		# print(y_predprob)

		# Rotate data on the discriminant coordinates
		p=lda.fit_transform(X,y)

		# Plot of LD components
		plt.scatter(x=p[:,0],y=p[:,1],c=y)
		plt.xlabel("LD1")
		plt.ylabel("LD2")
		# plt.show()
		# input()

		# Creates new data so we can predict it
		# Class variables
		newcr=[14,13,30,35,13]
		lda.predict([newcr])

		lda.predict_proba([newcr])

		# Define test and train sets
		X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=120)
		y_test_pred=lda.fit(X_train,y_train).predict(X_test)
		y_test-y_test_pred

		# KFolds algorithm
		# Tests validation between predicted and observed
		kf=KFold(n_splits=4,shuffle=True)
		kf.get_n_splits(X)
		test_preds=[]
		for train_index, test_index in kf.split(X):
		    X_train, X_test=X[train_index], X[test_index]
		    y_train, y_test=y[train_index], y[test_index]
		    lda.fit(X_train,y_train)
		    test_preds.append(y_test-lda.predict(X_test))
		# print("\n Test Predictions:")
		# print(test_preds)

		# Another what to do cross checks
		cross_val_score(estimator=lda,X=X,y=y,cv=kf)

		# This is the same as above but for quadratic discriminant analysis
		qda=QDA(store_covariance=True)
		analysis_QDA=qda.fit(X,y)
		y_pred=analysis_QDA.predict(X)
		print("\n Y Predictions:")
		print(y_pred)

		# Bit random?
		print("\n Y: Observed minus predicted")
		print(y-y_pred)

		y_predprob=analysis_QDA.predict_proba(X)
		print("\n Y Predicted Probability:")
		print(y_predprob)

		print("\n Y QDA Covariance:")
		print(qda.covariance_)

		print("\n Y QDA Rotations:")
		print(qda.rotations_)

		print("\n Y QDA Scalings:")
		print(qda.scalings_)

	def HelpMode(self):
		print("You're running in help mode!\n")
		print("Discriminant analysis is used to seperate groups as much as possible\n")
		print("Beware, there's no scaling in DA, unlike PCA, so large variances will dominate!\n")
		print("\n \t\tINPUT FILE:\t\t" + filename_path + "\n")
		print("Need to choose variables from the csv to preform analysis on. Need help?\nCHOOSE FROM:")
		print(df_DA_obj.DF.columns)
		print("\nNeed to choose variables (input_vars) that we use to discrimiant between the discriminant variables (discriminant_vars")
		input()
