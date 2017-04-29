print ("\n-------------------------------------\n")
print ("\n CREDIT CARD FRAUD DETECTION SYSTEM \n ")
print ("\n-------------------------------------\n")

print("\n Please wait, importing required libraries.\n")
print("\n IMPORTING PANDAS")
import pandas as pd 
print("\n IMPORTING NUMPY")
import numpy as np 
from sklearn.model_selection import train_test_split
print("\n IMPORTING MATPLOTLIB PYPLOT")
import matplotlib.pyplot as plt
print("\n IMPORTING SEABORN")
import seaborn as sns
print("\n IMPORTING SKLEARN METRICS")
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from itertools import cycle
print("\n IMPORTING CONFUSION MATRIX")
from sklearn.metrics import confusion_matrix
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange','black','pink'])

print("\n")
print("\n LOADING THE DATASET" )    
print("-----------------------\n")
df = pd.read_csv("creditcard.csv")
print("\n\n DATASET LOADED SUCCESSFULY")
print("--------------------------------\n")

def view_dataset():
    print("\n\n DATASET IS AS FOLLOWS")
    print("---------------------------\n")
    print(df)


def describe_dataset():
    print("\n\n\n")
    print("DESCRIPTION OF DATASET\n")
    print("-------------------------\n")
    dataset_describe=df.describe()
    print(dataset_describe)


def transaction_and_time():
    print("\n\n\n")
    print("Time Comparission across Fradulent and Normal Transactions\n")
    print("------------------------------------------------------------\n")
    op = int(input("1: View in Histogram\n2: View in Normal Text\n3: Go to main menu\n"))

    if op==1:
        f, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(12,4))
        bins = 50
        ax1.hist(df.Time[df.Class==1],bins=bins)
        ax1.set_title('Fraud')
        ax2.hist(df.Time[df.Class==0],bins=bins)
        ax2.set_title('Normal')
        plt.xlabel('Time(in seconds)')
        plt.ylabel('Number of Transactions')
        plt.show()

    elif op ==2:
        print("\nFraud")
        print("-----")
        time_accross_fraud=df.Time[df.Class==1].describe()
        print(time_accross_fraud)
        print("\n")
        print("\nNormal")
        print("-----")
        time_accross_normal=df.Time[df.Class==0].describe()
        print(time_accross_normal)
        return

    elif op ==3:
        pass

    else :
        print("\n INVALID OPTION")            
        pass

def amount_and_transactions():
        print("\n\n")
        print("Transactional AMOUNT Fradulent and Normal \n")
        print("-----------------------------------------\n")
        op = input("1: View in Histogram\n2: View in Normal Text\n3: Go to main menu\n")
        if op=='1':
            f,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(12,4))
            bins = 30
            ax1.hist(df.Amount[df.Class==1],bins=bins)
            ax1.set_title('Fraud')
            ax2.hist(df.Amount[df.Class==0],bins=bins)
            ax2.set_title('Normal')
            plt.xlabel('Amount ($)')
            plt.ylabel('Number of Transactions')
            plt.yscale('log')
            plt.show()
        elif op=='2':
            print('\nFraud')
            print("-----")
            print(df.Amount[df.Class==1].describe())    
            print('\nNormal')
            print("-----")
            print(df.Amount[df.Class==0].describe())
        elif op=='3':
            pass    
        else :
            print('Invalid Options')
            pass    

def time_and_amount():
    f,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(12,6))
    ax1.scatter(df.Time[df.Class==1],df.Amount[df.Class==1])
    ax1.set_title('Fraud')
    ax2.scatter(df.Time[df.Class==0],df.Amount[df.Class==0])
    ax2.set_title('Normal')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('AMount in Dollars')
    plt.show()    

def roc_curve_acc(Y_test, Y_pred,method):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color=next(colors),label='%s AUC = %0.3f'%(method, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    
def gaussian_nb(X_train,Y_train,X_test,Y_test):
    print("IMPORTING GaussianNB")
    from sklearn.naive_bayes import GaussianNB
    GNB=GaussianNB()
    GNB.fit(X_train, Y_train)
    Y_pred=GNB.predict(X_test)
    print("Gaussian NB report \n",classification_report(Y_pred,Y_test))
    print('*****roc_auc_score: %0.3f*****\n'% roc_auc_score(Y_pred,Y_test))
    roc_curve_acc(Y_test, Y_pred,'GNB')
    print("Gaussian NB confusion matrix \n",confusion_matrix(Y_pred,Y_test))

def random_forest(X_train,Y_train,X_test,Y_test):
    print("IMPORTING RandomForest")
    from sklearn.ensemble import RandomForestClassifier
    RF=RandomForestClassifier()
    RF.fit(X_train, Y_train)
    Y_pred=RF.predict(X_test)
    print("Random Forest Classifier Report \n", classification_report(Y_pred,Y_test))
    print("*****roc_auc_score: %0.3f*****\n"% roc_auc_score(Y_pred,Y_test))
    roc_curve_acc(Y_test, Y_pred,"RF")
    print("Random Forest Classfier confusion matrix \n",confusion_matrix(Y_pred,Y_test))    

def logistic_regression(X_train,Y_train,X_test,Y_test):
    print("IMPORTING Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    LR=LogisticRegression()
    LR.fit(X_train, Y_train)
    Y_pred=LR.predict(X_test)
    print("Logistic Regression report \n",classification_report(Y_pred,Y_test))
    print('*****roc_auc_score: %0.3f*****\n'% roc_auc_score(Y_pred,Y_test)) 
    roc_curve_acc(Y_test, Y_pred,'LR')
    print("Logistic Regression confusion matrix \n",confusion_matrix(Y_pred,Y_test))    
     
def algoriths():
    print("\n\n")
    print("Preparing DATASET...\n")
    print("Dropping features that have similar distribution b/w Fraudlent and Normal...\n")
    X=df[['V1','V2','V3','V4','V5','V6','V7','V9','V10','V11','V12','V14','V16','V17','V18','V19','V21']]
    Y=df["Class"]
    X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.10, random_state=1245)
    print("Select Any Algorithm")
    print("--------------------\n")
    op = input("1: Gaussian Naive Bayes(GaussianNB)\n2: Random Forest Classifier\n3: LogisticRegression\n4: Compare all algorithms\n5: Back to main menu\n")
    
    if op=='1':
       gaussian_nb(X_train,Y_train,X_test,Y_test)
       plt.show()
       
    elif op=='2':
       random_forest(X_train,Y_train,X_test,Y_test)
       plt.show()
         
    elif op=='3':
       logistic_regression(X_train,Y_train,X_test,Y_test)
       plt.show()
                
    elif op=='4':
        print("-------------------------------------------")
        gaussian_nb(X_train,Y_train,X_test,Y_test)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        random_forest(X_train,Y_train,X_test,Y_test)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        logistic_regression(X_train,Y_train,X_test,Y_test)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        plt.show()
    elif op=='5':
        pass         

def end():
    
    op = input("You have opted to exit. exit ? y/n")
    if op=='Y' or op=='y' or op=='yes' or op=='YES' or op =='Yes':
        exit()
    elif op=='N' or op=='n' or op=='No' or op =='no' or op=='NO' :
        pass
    else :
        print("INVALID OPTION.")    


while True:
    print("\n\n\n")
    print("*****MENU*****")
    print("--------------\n")
    print("1. View DataSet\n")
    print("2. Describe DataSet\n")
    print("3. Time Comparission across Fradulent and Normal Transactions (No: OF TRANSACTIONS vs TIME)\n")
    print("4. Transaction Amount Difference b/w Fraudlent and Normal (No: OF TRANSACTIONS vs AMOUNT) \n")
    print("5. Amount with Time Difference (AMOUNT vs TIME) \n")
    print("6. Analyse the Data using Algorithms \n")
    print("7. Exit \n")
    option = input("select any option : \n")
    if option=='1':
          view_dataset()
    if option=='2':
          describe_dataset()    
    if option=='3':
          transaction_and_time()
    elif option =='4':
        amount_and_transactions()    
    elif option=='5':
        time_and_amount()
    elif option=='6':
        algoriths()
    elif option=='7':
        end()        
    else :
        print("Invalid Option")            



    



