# Client_Model_Deplyoment
Agenda : To build a model and deploy in streanlit app, so the user will get to know thee eligibility for insuance policy.

Two programs to be written first one to train and test the model and second program is the main program where you deploy the the trained model in streamlit app
creating webpage using the streamlit magic command.

Program 1 :
Import the required packages.
Read the data set.
Check the missing values and fill the missing values accordingly.
Check the outliers using boxplot and remove outliers using capping methode.
Select the features and target.
Train Test split.
Apply transformation for continuous column's.
Define your model with logistic regression cv.
Train the model and test the model.
Find the prediction.
Save the model in pickel format, open the file and dump you model.
Dump the standard scler also because user information may not be in ascaled format.
Download the .pkl format file and then past in your directory(know your directory by anaconda prompt).

Program 2 :
Import the required packages.
Now define the user input parameters accordingly with the streamlit app commands(it's to creat web page).
user gives input depending on the predict probability the user will get to know his eligibility for insurance policy.
Now dowload this in exectable format and paste in the directory.

Anaconda prompt : Run the file(streamlit run file.py)
Initally it asks your mail id later after entering mail id it will redirect to streamlit app where user can enter the input and check his eligibility for insurance policy.
