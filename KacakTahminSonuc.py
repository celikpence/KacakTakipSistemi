# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:39:39 2018

@author: mustafa.celikpence
"""
from __future__ import print_function
#1. Data Load

import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, accuracy_score, recall_score, confusion_matrix, mean_squared_error, r2_score, f1_score, precision_score

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest

import pyswarms as ps
import statsmodels.api as sm
import seaborn as sns
import io


from sklearn import datasets, linear_model


DefaultTestSize=0.33

# Feature Selection RFE Tanımlar ----------------------------------------------

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# BPSO Tanımlar ---------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=165, n_classes=3,
                           n_informative=4, n_redundant=1, n_repeated=2,
                           random_state=1)


# Create an instance of the classifier
classifier = linear_model.LogisticRegression()
#classifier = RandomForestClassifier(n_estimators=10, criterion = 'gini', random_state=250)

# Define objective function

def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 15
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_Subset = XStd_Test
    else:
        X_Subset = XStd_Test[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_Subset, y_test.values.ravel())
    P = (classifier.predict(X_Subset) == y_test['KacakDur']).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_Subset.shape[1] / total_features)))

    return j

def f(x, alpha=0.90):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


def getBPSOSelectedColumnNames():
    i = 0
    ColumnNames = []
    while i < len(BPSOColumnIndexesFull):
            ColumnNames.append(KacakTahmin.columns[i+1])  # İlk kolon Tesisat No sonradan çıkarıldığı için her zaman +1
            i += 1
    return ColumnNames
    
# BPSO Tanımlar Sonu ----------------------------------------------------------

# Precision Recall Curve------------------------------------------------------

def DrawPrecisionRecallCurve(ShortAlgorithmName, logr, line_color, showgraph, lastdraw):

    from sklearn.metrics import average_precision_score
    from sklearn.metrics import plot_precision_recall_curve
    
    average_precision = average_precision_score(y_test, y_pred)
    
    print(ShortAlgorithmName + ' Average Precision-Recall score: {0:0.2f}'.format(average_precision))
    
    if showgraph:   
        disp = plot_precision_recall_curve(logr, XStd_Test, y_test, ax = plt.gca(), name=ShortAlgorithmName, color=line_color)
        disp.ax_.legend(loc='upper right')
        if lastdraw:
            disp.ax_.set_title('Precision Recall Curve')
  
# ROC Curve  -----------------------------------------------------

def DrawROCCurve(AlgorithmName, ML_Algorithm, draw_color, firstdraw, drawgraph, lastdraw):
    import sklearn.metrics as metrics
    # calculate the fpr and tpr for all thresholds of the classification
    probs = ML_Algorithm.predict_proba(XStd_Test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print( AlgorithmName +' AUC = %0.2f' % roc_auc)
    
    if drawgraph:
        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label = AlgorithmName +' AUC = %0.2f' % roc_auc, color=draw_color)    
        plt.legend(loc = 'lower right', prop={'size': 10})
        if firstdraw:
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
        if lastdraw:
            plt.show()  

# Karmasiklik matrisi ----------------------------------------------------------

def CalcConfMatrix(ML_AlgorithmName, ytest, ypred, showgraph):
    cm = confusion_matrix(ytest,ypred)
    print(ML_AlgorithmName + " Confusion Matrix")
    print(cm)
    
    accuracy = accuracy_score(ytest, ypred)
    accuracy = round(accuracy, 4)
    
    recall  = recall_score(ytest, ypred) # recall = tp / (tp + fn)  Kaçak Doğru / Toplam Kaçak
    recall = round(recall, 4)
    
    precision = precision_score(ytest, ypred)
    precision = round(precision, 4)
    
    f1 = f1_score(ytest, ypred)
    f1 = round(f1, 4)
     
     
    print(ML_AlgorithmName + " Accuracy: %.2f%%" % (accuracy * 100.0))
    print(ML_AlgorithmName + " Recall: %.2f%%" % (recall * 100.0))
    print(ML_AlgorithmName + " Precision: %.2f%%" % (precision * 100.0))
    print(ML_AlgorithmName + " F1 Score: %.2f%%" % (f1 * 100.0))
    
    if showgraph: # Confusion Matrix Grafik
        plt.figure()
        sns.heatmap(cm, annot=True, fmt=".0f")
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.suptitle(ML_AlgorithmName + ' Recall Score: {0}'.format(recall), size = 14)
        plt.title(ML_AlgorithmName + ' Accuracy Score: {0}'.format(accuracy), size = 14)
        #plt.title("Confusion Matrix - " + ML_AlgorithmName, size = 14)
        plt.show()
  
    
# 10 K-Fold Validation (k-katlamali capraz dogrulama) -------------------------
        
def CalcKFold(ML_AlgorithmName, ML_Algorithm, K_Value):
    from sklearn.model_selection import cross_val_score, cross_validate
    #Estimator'ımıza oluşturduğumuz modeli gönderiyoruz, cv : kaç parçaya bölüneceğini belirler.Genellikle 10 alınır.
    accuracies = cross_val_score(estimator = ML_Algorithm, X = XStd_Train, y = y_train.values.ravel(), cv = K_Value)
    print(ML_AlgorithmName + " K-Fold Doğruluk : %", round(accuracies.mean()*100,2))
    print(ML_AlgorithmName + " K-Fold Standart Sapma: %",round(accuracies.std()*100,2))

    scores = cross_validate(ML_Algorithm, X = XStd_Train, y = y_train, cv = K_Value, scoring=('accuracy', 'average_precision', 'f1', 'recall', 'roc_auc'), return_train_score=True)  
    
    # Karmaşılık Matrisi Sonuçları
    print(ML_AlgorithmName + " k = " + str(K_Value))
    print("Accuracy Score : " + str("%.4f" % scores['test_accuracy'].mean()))
    print("Recall Score : " + str("%.4f" % scores['test_recall'].mean()))
    print("Precision Score : " + str("%.4f" % scores['test_average_precision'].mean()))
    print("F1 Score : " + str("%.4f" % scores['test_f1'].mean()))   
    print("ROC_AUC: " + str(scores['test_roc_auc'].mean()))
    
    
def ShowRMSE(ML_AlgorithmName, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(ML_AlgorithmName + " RMSE: %f" % (rmse))   

def UpdateStrFormat(ColumnValue):
    ColumnValue = ColumnValue

def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return mse
    
def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return mse
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error

# Console Temizleme
try:
    from IPython import get_ipython
    get_ipython().magic('clear')    
except:
    pass
  
# MSSQL Veritabanından Verileri Alma
'''connection = pyodbc.connect('Driver={SQL Server};'
                                'Server=localhost\SQLEXPRESS;'
                                'Database=DBKACAK;Trusted_Connection=yes;'
'''
connection = pyodbc.connect('Driver={SQL Server}; Database=DBKACAK;  ')
# MSSQL bağlantı bilgileri güvenlik gereği paylaşılmamıştır.
                                

KacakSQL = 'Select TesisatNo, AboneGrubu, AGOG, OlcumDurumu, Tarife, Sektor, Il, Ilce, Koy, SayacModeli, BaglantiDurumu, KuruluGuc, OrtalamaTuketim, ' \
'OrtalamaGunlukTuketimAy1, OrtalamaGunlukTuketimAy2, OrtalamaGunlukTuketimAy3, OrtalamaGunlukTuketimAy4, OrtalamaGunlukTuketimAy5, OrtalamaGunlukTuketimAy6,' \
'OrtalamaGunlukTuketimAy7, OrtalamaGunlukTuketimAy8, OrtalamaGunlukTuketimAy9, OrtalamaGunlukTuketimAy10, OrtalamaGunlukTuketimAy11, OrtalamaGunlukTuketimAy12,' \
'MaxDemandAy1, MaxDemandAy2, MaxDemandAy3, MaxDemandAy4, MaxDemandAy5, MaxDemandAy6, MaxDemandAy7, MaxDemandAy8, MaxDemandAy9, MaxDemandAy10, MaxDemandAy11, MaxDemandAy12,' \
'OrtalamaAkim1Ay1, OrtalamaAkim1Ay2, OrtalamaAkim1Ay3, OrtalamaAkim1Ay4, OrtalamaAkim1Ay5, OrtalamaAkim1Ay6, OrtalamaAkim1Ay7, OrtalamaAkim1Ay8, OrtalamaAkim1Ay9, OrtalamaAkim1Ay10, OrtalamaAkim1Ay11, OrtalamaAkim1Ay12,' \
'OrtalamaAkim2Ay1, OrtalamaAkim2Ay2, OrtalamaAkim2Ay3, OrtalamaAkim2Ay4, OrtalamaAkim2Ay5, OrtalamaAkim2Ay6, OrtalamaAkim2Ay7, OrtalamaAkim2Ay8, OrtalamaAkim2Ay9, OrtalamaAkim2Ay10, OrtalamaAkim2Ay11, OrtalamaAkim2Ay12,' \
'OrtalamaAkim3Ay1, OrtalamaAkim3Ay2, OrtalamaAkim3Ay3, OrtalamaAkim3Ay4, OrtalamaAkim3Ay5, OrtalamaAkim3Ay6, OrtalamaAkim3Ay7, OrtalamaAkim3Ay8, OrtalamaAkim3Ay9, OrtalamaAkim3Ay10, OrtalamaAkim3Ay11, OrtalamaAkim3Ay12,' \
'OrtalamaGerilim1Ay1, OrtalamaGerilim1Ay2, OrtalamaGerilim1Ay3, OrtalamaGerilim1Ay4, OrtalamaGerilim1Ay5, OrtalamaGerilim1Ay6, OrtalamaGerilim1Ay7, OrtalamaGerilim1Ay8, OrtalamaGerilim1Ay9, OrtalamaGerilim1Ay10, OrtalamaGerilim1Ay11, OrtalamaGerilim1Ay12,' \
'OrtalamaGerilim2Ay1, OrtalamaGerilim2Ay2, OrtalamaGerilim2Ay3, OrtalamaGerilim2Ay4, OrtalamaGerilim2Ay5, OrtalamaGerilim2Ay6, OrtalamaGerilim2Ay7, OrtalamaGerilim2Ay8, OrtalamaGerilim2Ay9, OrtalamaGerilim2Ay10, OrtalamaGerilim2Ay11, OrtalamaGerilim2Ay12,' \
'OrtalamaGerilim3Ay1, OrtalamaGerilim3Ay2, OrtalamaGerilim3Ay3, OrtalamaGerilim3Ay4, OrtalamaGerilim3Ay5, OrtalamaGerilim3Ay6, OrtalamaGerilim3Ay7, OrtalamaGerilim3Ay8, OrtalamaGerilim3Ay9, OrtalamaGerilim3Ay10, OrtalamaGerilim3Ay11, OrtalamaGerilim3Ay12,' \
'CosF1OrtalamaSapmaAy1, CosF1OrtalamaSapmaAy2, CosF1OrtalamaSapmaAy3, CosF1OrtalamaSapmaAy4, CosF1OrtalamaSapmaAy5, CosF1OrtalamaSapmaAy6, CosF1OrtalamaSapmaAy7, CosF1OrtalamaSapmaAy8, CosF1OrtalamaSapmaAy9, CosF1OrtalamaSapmaAy10, CosF1OrtalamaSapmaAy11, CosF1OrtalamaSapmaAy12,' \
'CosF2OrtalamaSapmaAy1, CosF2OrtalamaSapmaAy2, CosF2OrtalamaSapmaAy3, CosF2OrtalamaSapmaAy4, CosF2OrtalamaSapmaAy5, CosF2OrtalamaSapmaAy6, CosF2OrtalamaSapmaAy7, CosF2OrtalamaSapmaAy8, CosF2OrtalamaSapmaAy9, CosF2OrtalamaSapmaAy10, CosF2OrtalamaSapmaAy11, CosF2OrtalamaSapmaAy12,' \
'CosF3OrtalamaSapmaAy1, CosF3OrtalamaSapmaAy2, CosF3OrtalamaSapmaAy3, CosF3OrtalamaSapmaAy4, CosF3OrtalamaSapmaAy5, CosF3OrtalamaSapmaAy6, CosF3OrtalamaSapmaAy7, CosF3OrtalamaSapmaAy8, CosF3OrtalamaSapmaAy9, CosF3OrtalamaSapmaAy10, CosF3OrtalamaSapmaAy11, CosF3OrtalamaSapmaAy12,' \
'Kacak KacakDur From TheftsSample '

AboneGrupSQL = 'SELECT Distinct AboneGrubu  FROM TheftsSample Group By AboneGrubu'
AboneGrupColumns = pd.read_sql(AboneGrupSQL , connection)

SayacModeliSQL = 'SELECT Distinct SayacModeli  FROM TheftsSample Group By SayacModeli'
SayacModeliColumns = pd.read_sql(SayacModeliSQL , connection)

KacakTahminVerileri = pd.read_sql(KacakSQL, connection)

#connection.close()

# ----------------------------------------------------------------------------
#1 Data Reprocessing
# ----------------------------------------------------------------------------

#1.1 Encoder : Kategorik -> Numeric


TesisatNo = KacakTahminVerileri['TesisatNo']

AboneGrubu = KacakTahminVerileri.iloc[:,1:2].values #KacakTahminVerileri['AboneGrubu']
OlcumDurumu = KacakTahminVerileri.iloc[:,3:4].values 
# Tarife = KacakTahminVerileri.iloc[:,3:4].values
# Sektor = KacakTahminVerileri.iloc[:,5:6].values
Il = KacakTahminVerileri.iloc[:,6:7].values
#Ilce = KacakTahminVerileri.iloc[:,6:7].values
SayacModeli = KacakTahminVerileri.iloc[:,9:10].values
#BaglantiDurumu = KacakTahminVerileri.iloc[:,11:12].values

#KacakSonuc = KacakTahminVerileri.iloc[:,-1].values
KacakSonuc = KacakTahminVerileri.loc[:,['TesisatNo','KacakDur']].values
DFKacakSonuc = pd.DataFrame(data = KacakSonuc, index = range(KacakSonuc.shape[0]), columns=['TesisatNo','KacakDur'])

# ----------------------------------------------------------------------------
#1.1.1 Label Encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

AboneGrubu[:,0] = le.fit_transform(AboneGrubu[:,0])
OlcumDurumu[:,0] = le.fit_transform(OlcumDurumu[:,0])
# Tarife[:,0] = le.fit_transform(Tarife[:,0])
# Sektor[:,0] = le.fit_transform(Sektor[:,0])
Il[:,0] = le.fit_transform(Il[:,0])
# Ilce[:,0] = le.fit_transform(Ilce[:,0])
SayacModeli[:,0] = le.fit_transform(SayacModeli[:,0])
#BaglantiDurumu[:,0] = le.fit_transform(BaglantiDurumu[:,0])

# ----------------------------------------------------------------------------
#1.1.2 One Hot Encoding

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
#ohe = OneHotEncoder(categorical_features='all')
ohe = OneHotEncoder(categories='auto')

AboneGrubu = ohe.fit_transform(AboneGrubu).toarray()
OlcumDurumu = ohe.fit_transform(OlcumDurumu).toarray()
# Tarife = ohe.fit_transform(Tarife).toarray()
# Sektor = ohe.fit_transform(Sektor).toarray()
Il = ohe.fit_transform(Il).toarray()
#Ilce = ohe.fit_transform(Ilce).toarray()
SayacModeli = ohe.fit_transform(SayacModeli).toarray()
#BaglantiDurumu = ohe.fit_transform(BaglantiDurumu).toarray()

#PartAboneGruplari = pd.DataFrame(data = AboneGrubu, index = range(AboneGrubu.shape[0]), columns=[AboneGrupColumns.AboneGrubu.values])
#PartAboneGruplari = pd.DataFrame(data = AboneGrubu, index = range(AboneGrubu.shape[0]), columns=['RESMİ DAİRELER','KITLER','CAMİ - İBADETHANE','HİZMET BİNALARI','ARITMA TESİSLERİ','SANAYİ','TİCARETHANE','BELEDİYELER','ÜRETİCİLER','ICME SUYU','OSB/JEOTERMAL','ŞANTİYE','HAYIR KURUMLARI','AYDINLATMA','KARAYOLLARI+PARK AYDINLATMA','MESKEN','TARIMSAL SULAMA'] )
PartAboneGruplari = pd.DataFrame(data = AboneGrubu, index = range(AboneGrubu.shape[0]), columns=['RESMİ DAİRELER','KITLER','CAMİ - İBADETHANE','HİZMET BİNALARI','ARITMA TESİSLERİ','TİCARETHANE','BELEDİYELER','ÜRETİCİLER','ICME SUYU','ŞANTİYE','HAYIR KURUMLARI','KARAYOLLARI+PARK AYDINLATMA','MESKEN','TARIMSAL SULAMA'] )
PartOlcumDurumlari = pd.DataFrame(data = OlcumDurumu, index = range(OlcumDurumu.shape[0]), columns=['Diğer','Sekonder','Primer'] )
PartIller = pd.DataFrame(data = Il, index = range(Il.shape[0]), columns=['ŞIRNAK','DİYARBAKIR','ŞANLIURFA','SİİRT','MARDİN','BATMAN',])
#PartSayacModelleri = pd.DataFrame(data = SayacModeli, index = range(SayacModeli.shape[0]), columns=['VIK','LND','ESI','LUN','KHL','ELM','LNO','AEL','MSY','YOK'])
#PartSayacModelleri = pd.DataFrame(data = SayacModeli, index = range(SayacModeli.shape[0]), columns=['VIK','LND','ESI','LUN','ELM','LNO','AEL','MSY','YOK'])
PartSayacModelleri = pd.DataFrame(data = SayacModeli, index = range(SayacModeli.shape[0]), columns=[SayacModeliColumns.SayacModeli.values])
PartKoy = KacakTahminVerileri['Koy']

# ----------------------------------------------------------------------------
# 1.2 DataFrame partları birlestirme islemi

UniPart = pd.concat([PartAboneGruplari, PartOlcumDurumlari],axis=1)
UniPart = pd.concat([TesisatNo, UniPart],axis=1)
#UniPart = pd.concat([TesisatNo.toarray(), UniPart],axis=1)
UniPart = pd.concat([UniPart, PartIller],axis=1)
UniPart = pd.concat([UniPart, PartKoy],axis=1)
UniPart1 = pd.concat([UniPart, PartSayacModelleri],axis=1)
UniPart2 = KacakTahminVerileri.loc[:,'KuruluGuc':'CosF3OrtalamaSapmaAy12']
KacakTahmin =  pd.concat([UniPart1, UniPart2],axis=1)


# ParemetreAdlari = KacakTahmin.columns.tolist()

# ----------------------------------------------------------------------------
# 2 Verilerin Egitim ve Test icin bolunmesi
# ----------------------------------------------------------------------------


#from sklearn.model_selection import train_test_split
#XStd_Train, XStd_Test, y_train, y_test = train_test_split(KacakTahmin,KacakSonuc,test_size=0.33,random_state=0)


# ----------------------------------------------------------------------------
# X Train Test Tesisatlarının DB ye yazılması
# ----------------------------------------------------------------------------

'''
for index, row in XStd_Train.iterrows():   
   cur = connection.cursor()
   strInsSQL =  'INSERT INTO TesisatXTrainSample (TesisatNo) values ( {} )'.format(row["TesisatNo"])
   cur.execute(strInsSQL)
   cur.close()
   connection.commit()
'''

'''
for index, row in XStd_Test.iterrows():   
   cur = connection.cursor()
   strInsSQL =  'INSERT INTO TesisatXTestSample (TesisatNo) values ( {} )'.format(row["TesisatNo"])
   cur.execute(strInsSQL)
   cur.close()
   connection.commit()
'''

y_test = np.load('y_testsabit.npy')
XStd_Test = np.load('XStd_Testsabit.npy')

y_test = pd.DataFrame(data = y_test, index = range(y_test.shape[0]), columns=['TesisatNo','KacakDur'])
XStd_Test  = pd.DataFrame(data = XStd_Test , index = range(XStd_Test.shape[0]), columns = KacakTahmin.columns)
#y_train = pd.DataFrame(data = y_train, index = range(y_train.shape[0]), columns=['TesisatNo','KacakDur'])

# ----------------------------------------------------------------------------
# Y Train Test Tesisatlarının DB ye yazılması
# ----------------------------------------------------------------------------

#Dataframe çevirme işlemi

#y_train = pd.DataFrame(data = y_train, index = range(y_train.shape[0]), columns=['TesisatNo','KacakDur'])
#y_test = pd.DataFrame(data = y_test, index = range(y_test.shape[0]), columns=['TesisatNo','KacakDur'])

'''
for index, row in y_train.iterrows():   
   cur = connection.cursor()
   strInsSQL =  'INSERT INTO TesisatYTrainSample (TesisatNo,KacakDur) values ( {0},{1} )'.format(row["TesisatNo"],row["KacakDur"])
   cur.execute(strInsSQL)
   cur.close()
   connection.commit()

for index, row in y_test.iterrows():   
   cur = connection.cursor()
   strInsSQL =  'INSERT INTO TesisatYTestSample (TesisatNo,KacakDur) values ( {0},{1} )'.format(row["TesisatNo"],row["KacakDur"])
   cur.execute(strInsSQL)
   cur.close()
   connection.commit()
'''

'''
np.save('XStd_Trainsabit.npy', XStd_Train)
np.save('XStd_Testsabit.npy', XStd_Test)
np.save('y_trainsabit.npy', y_train)
np.save('y_testsabit.npy', y_test)
'''

LeftJoinXTestData = pd.merge(KacakTahmin, XStd_Test, on='TesisatNo', how='left',  suffixes=('', '_2'))
Filtered_DfX = LeftJoinXTestData[LeftJoinXTestData['KuruluGuc_2'].isnull()]  # Test datası ile eşleşmeyen kayıtlar - yani null olan kayıtlar(Training Veri Seti) alınıyor
XStd_Train = Filtered_DfX.dropna(axis=1)  # null içeren sütunlar yani left joinde eşleşmeyen kayıltardaki mükerrer kolonlar siliniyor
 
#LeftJoin.dropna(subset=['KuruluGuc_2'])
#aaa = LeftJoin.where(LeftJoin['KuruluGuc_2']=="nan")

LeftJoinYTestData = pd.merge(DFKacakSonuc, y_test, on='TesisatNo', how='left',  suffixes=('', '_2'))
Filtered_DfY = LeftJoinYTestData[LeftJoinYTestData['KacakDur_2'].isnull()]  # Test datası ile eşleşmeyen kayıtlar - yani null olan kayıtlar(Training Veri Seti) alınıyor
y_train = Filtered_DfY.dropna(axis=1)  # null içeren sütunlar yani left joinde eşleşmeyen kayıltardaki mükerrer kolonlar siliniyor


'''
DFKacakSonuc['TesisatNo2'] = y_test['TesisatNo'] # KacakSonuc datasetinde test için ayrılan tesisatlar, TesisatNo2 alanına yazılıyor
indexNames = DFKacakSonuc[ (DFKacakSonuc['TesisatNo'] == DFKacakSonuc['TesisatNo2'])].index
DFKacakSonuc.drop(indexNames , inplace=True)
DFKacakSonuc1 = DFKacakSonuc.drop('TesisatNo2', axis='columns')
y_train = DFKacakSonuc1
'''

#y_test111 = np.load('y_testsabit.npy')

XStd_Train = XStd_Train.iloc[:, 1:]  # TesisatNo alanı çıkarılıyor
XStd_Test = XStd_Test.iloc[:, 1:]  # TesisatNo alanı çıkarılıyor
y_train = y_train.iloc[:, 1:]  # TesisatNo alanı çıkarılıyor
y_test = y_test.iloc[:, 1:]  # TesisatNo alanı çıkarılıyor

connection.close()

# shuffle parametresi için model_selection içindeki train_test_split kullanılması
#from sklearn.model_selection import train_test_split
#XStd_Train, XStd_Test, y_train, y_test = train_test_split(KacakTahmin,KacakSonuc,test_size=DefaultTestSize,random_state=0)
# XStd_Train, XStd_Test, y_train, y_test = train_test_split(KacakTahmin,KacakSonuc,test_size=0.33,random_state=0, shuffle=False)

#np.save('XStd_Trainsabit.npy', XStd_Train)
#XStd_Train = np.load('XStd_Trainsabit.npy')

#np.save('y_trainsabit.npy', y_train)
#y_train = np.load('y_trainsabit.npy')

#np.save('XStd_Testsabit.npy', XStd_Test)
#XStd_Test = np.load('XStd_Testsabit.npy')

#np.save('y_testsabit.npy', y_test)
#y_test = np.load('y_testsabit.npy')



#col_mask=XStd_Train.isnull().any(axis=0)
#col_mask2=XStd_Train.isnull().any(axis=0)

# 2.1 Verilerin olceklenmesi

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

XStd_Train = sc.fit_transform(XStd_Train)
XStd_Test = sc.fit_transform(XStd_Test)



# ----------------------------------------------------------------------------

# 3 Sınıflandırma Algoritmaları

# ----------------------------------------------------------------------------

k_Fold = 10

# 3.1 Logistic Regression

AlgorithmName = 'Logistic Regression'
ShortAlgorithmName = 'LR'

#from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=250, max_iter=500)

logr.fit(XStd_Train,y_train.values.ravel())
y_pred = logr.predict(XStd_Test)

ShowRMSE(AlgorithmName, y_test, y_pred)
CalcConfMatrix(ShortAlgorithmName, y_test, y_pred, 1)
CalcKFold(AlgorithmName, logr, k_Fold)
DrawROCCurve(ShortAlgorithmName, logr, 'red', 1, 1, 0)
DrawPrecisionRecallCurve(ShortAlgorithmName, logr, 'darkorange', 1, 0)

# ----------------------------------------------------------------------------

# 3.2 KNN

AlgorithmName = 'KNN'
ShortAlgorithmName = 'KNN'
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(XStd_Train,y_train.values.ravel())
y_pred = knn.predict(XStd_Test)

ShowRMSE(AlgorithmName, y_test, y_pred)
CalcConfMatrix(AlgorithmName, y_test, y_pred, 1)
CalcKFold(AlgorithmName, knn, k_Fold)
DrawROCCurve(AlgorithmName, knn, 'orange', 0, 1, 0)
DrawPrecisionRecallCurve(AlgorithmName, knn, 'teal', 1, 0)
# ----------------------------------------------------------------------------

# 3. SVC (SVM classifier)

AlgorithmName = 'SVM'
ShortAlgorithmName = 'SVM'

from sklearn.svm import SVC
svc = SVC(kernel='poly', probability=True, degree=4, class_weight='balanced', coef0=3, decision_function_shape='ovo', gamma='scale', random_state=250)
svc.fit(XStd_Train,y_train.values.ravel())
y_pred = svc.predict(XStd_Test)

ShowRMSE(AlgorithmName, y_test, y_pred)
CalcConfMatrix(AlgorithmName, y_test, y_pred, 1)
CalcKFold(AlgorithmName, svc, k_Fold)
DrawROCCurve(AlgorithmName, svc, 'purple', 0, 1, 0)
DrawPrecisionRecallCurve(AlgorithmName, svc, 'cornflowerblue', 1, 0)

# ----------------------------------------------------------------------------


AlgorithmName = 'SVM GridSearch'

from sklearn.svm import SVC
svc = SVC()

parameters = {'kernel': ['linear','poly', 'rbf', 'sigmoid'], 
             'degree': [2, 3, 4], 
              'gamma': ['scale', 'auto'],              
              'coef0': [0.0, 0.5, 1, 2, 3, 5, 10],
             'class_weight': ['balanced', 'dict'],
             'decision_function_shape': ['ovo', 'ovr'],
             'random_state': [250]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
gs = GridSearchCV(svc, parameters, scoring=acc_scorer)
#grid_obj = GridSearchCV(estimator=svc, param_grid=parameters, scoring=acc_scorer, cv=10, n_jobs=1)
grid_obj = gs.fit(XStd_Train, y_train.values.ravel())

# Set the clf to the best combination of parameters
svc = grid_obj.best_estimator_
best_parameters = grid_obj.best_params_  
print('En iyi parametreler')
print(best_parameters)  

print("Best score: %0.3f" % grid_obj.best_score_)
print("Best parameters set:")
best_parameters=grid_obj.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


svc.fit(XStd_Train, y_train.values.ravel())
y_pred2 = svc.predict(XStd_Test)

ShowRMSE(AlgorithmName, y_test, y_pred)
CalcConfMatrix(AlgorithmName, y_test, y_pred2, 0)
CalcKFold(AlgorithmName, svc, 10)

#--------------------------------------------------------------------------------------
'''
#Train Error Test Error

#train_error, test_error = calc_metrics(XStd_Train, y_train, XStd_Test, y_test, svc)
#train_error, test_error = round(train_error, 3), round(test_error, 3)

#print('train error: {} | test error: {}'.format(train_error, test_error))
#print('train/test: {}'.format(round(test_error/train_error, 1)))
'''
# --------------------------------------------------------------------------------------

# 4. Decision Tree

AlgorithmName = 'Decision Tree'
ShortAlgorithmName = 'DT'

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy', random_state=250)

dtc.fit(XStd_Train,y_train.values.ravel())
y_pred = dtc.predict(XStd_Test)

ShowRMSE(ShortAlgorithmName, y_test, y_pred)
CalcConfMatrix(ShortAlgorithmName, y_test, y_pred, 1)
CalcKFold(AlgorithmName, dtc, k_Fold)
DrawROCCurve(ShortAlgorithmName, dtc, 'green', 0, 1, 0)
DrawPrecisionRecallCurve(ShortAlgorithmName, dtc, 'gold', 1, 0)

# ----------------------------------------------------------------------------

# 5. Random Forest

AlgorithmName = 'Random Forest'
ShortAlgorithmName = 'RF'

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=9, criterion = 'entropy', max_depth=10 , random_state=250, min_samples_leaf=5, min_samples_split=2)

rfc.fit(XStd_Train,y_train.values.ravel())
y_pred = rfc.predict(XStd_Test)
classifier = rfc

ShowRMSE(AlgorithmName, y_test, y_pred)
CalcConfMatrix(ShortAlgorithmName, y_test, y_pred, 1)
CalcKFold(AlgorithmName, rfc, k_Fold, 1)
DrawROCCurve(ShortAlgorithmName, rfc, 'blue', 0, 1, 0)
DrawPrecisionRecallCurve(ShortAlgorithmName, classifier, 'red', 1, 0)

#------------------------------------------------------------------------

# 5.1. Random Forest GridSearch 
# 

AlgorithmName = 'Random Forest GridSearch'

from sklearn.ensemble import RandomForestClassifier
rfcGrd = RandomForestClassifier(random_state=250)

parameters = {'n_estimators': [2, 4, 6, 8, 9, 10], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 4, 5, 8, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,2,4,5,8],
              'random_state' : [250]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
gs = GridSearchCV(rfcGrd, parameters, scoring=acc_scorer)

#grid_obj = GridSearchCV(estimator=rfcGrd, param_grid=parameters, scoring=acc_scorer, cv=10, n_jobs=1)
grid_obj = gs.fit(XStd_Train,  y_train.values.ravel())

# Set the clf to the best combination of parameters
rfcGrd = grid_obj.best_estimator_
best_parameters = grid_obj.best_params_  
print('En iyi parametreler')
print(best_parameters)  

print("Best score: %0.3f" % grid_obj.best_score_)
print("Best parameters set:")
best_parameters=grid_obj.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

rfcGrd.fit(XStd_Train, y_train.values.ravel())
y_pred2 = rfcGrd.predict(XStd_Test)

ShowRMSE(AlgorithmName, y_test, y_pred)
CalcConfMatrix(AlgorithmName, y_test, y_pred2, 0)
CalcKFold(AlgorithmName, rfcGrd, 10)

# ----------------------------------------------------------------------------

# 6. XGBoost Model

AlgorithmName = 'XGBoost'
ShortAlgorithmName = 'XGB'
from xgboost import XGBClassifier

# fit model no training data
model = XGBClassifier(booster='gbtree', eval_metric='mlogloss')   #gblinear, gbtree

#model = XGBClassifier(booster='gbtree',
#learning_rate =0.1,
# n_estimators=1000,
# max_depth=5,
# min_child_weight=1,
# gamma=0,
# subsample=0.8,
# colsample_bytree=0.8,
# objective= 'binary:logistic',
# nthread=4,
# scale_pos_weight=1,
# seed=27)
 
model.fit(XStd_Train,y_train)

# make predictions for test data
y_pred = model.predict(XStd_Test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

CalcConfMatrix(AlgorithmName, y_test, y_pred, 1)
CalcKFold(AlgorithmName, model, k_Fold)
DrawROCCurve(ShortAlgorithmName, model, 'yellow', 0, 1, 1)
DrawPrecisionRecallCurve(ShortAlgorithmName, model, 'maroon', 1, 1)

# ----------------------------------------------------------------------------

# 7. RFE - Rekürsif Future Elimination

SelectedFeatureCount = 70
model = LogisticRegression(random_state=250, solver='lbfgs',multi_class='multinomial').fit(XStd_Train, y_train.values.ravel())

rfe = RFE(model, SelectedFeatureCount)
fit = rfe.fit(XStd_Train, y_train.values.ravel())

print( fit.n_features_) 
print("Selected Features: %s"% fit.support_) 
print("Feature Ranking: %s"% fit.ranking_)

RFEColumnIndexes = rfe.get_support(True)

# -------------
# Seçilen parametreler ile tekrar algoritma çalıştırma

#from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=250, max_iter=500)

logr.fit(XStd_Train,y_train.values.ravel())
y_pred = logr.predict(XStd_Test)

RFESelected = XStd_Train[:,RFEColumnIndexes]  # RFE ile seçilen kolonlardan yeni bir dizi oluşturuluyor
RFESelectedTest = XStd_Test[:,RFEColumnIndexes]

# Kolon isimlerinin görülebilmesi için dataframe oluşturuldu
DF_RFESelected = pd.DataFrame(data = RFESelected, index = range(RFESelected.shape[0]), columns=KacakTahmin.iloc[:,RFEColumnIndexes].columns)

#RFESelected.columns =  KacakTahmin.iloc[:,RFEColumnIndexes].columns

# -------------
# 7.1. Random Forest RFE Optimum Parameter ile Tahmin

XStd_Train = RFESelected
XStd_Test = RFESelectedTest

AlgorithmName = 'LR RFE (Parametre Sayısı : ' + str(SelectedFeatureCount) + ')'

logr2 = LogisticRegression(random_state=250, max_iter=500)

logr2.fit(XStd_Train,y_train)
y_pred2 = logr2.predict(XStd_Test)

ShowRMSE(AlgorithmName, y_test, y_pred2)
CalcConfMatrix(AlgorithmName, y_test, y_pred2, 1)
CalcKFold(AlgorithmName, logr2, 10)

#print(accuracy_score(y_test, y_pred2))

#-------------------------------------------
# 7.3 Train Error Test Error
'''
train_error, test_error = calc_metrics(XStd_Train, y_train, XStd_Test, y_test, rfc)
train_error, test_error = round(train_error, 3), round(test_error, 3)

print('train error: {} | test error: {}'.format(train_error, test_error))
print('train/test: {}'.format(round(test_error/train_error, 1)))
'''

# ----------------------------------------------------------------------------
# 8. Feature Importance - Extra Tree Classifier

from sklearn.ensemble import ExtraTreesClassifier

# fit an Extra Trees model to the data

model = ExtraTreesClassifier(random_state=420)
model.fit(KacakTahmin, KacakSonuc)

# display the relative importance of each attribute

FeatureImportance = model.feature_importances_
#print(FeatureImportance)

indices = np.argsort(FeatureImportance)[::-1]

# Print the feature ranking
#print("Feature ranking:")

#for f in range(KacakTahmin.shape[1]):
#   print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImportance[indices[f]]))

SelectedParameterNum = 20  # SEçilecek en iyi parametrelerin sayısı 

# model=ExtraTreesClassifier()
# model=RFECV(model,cv=3)
# model.fit(features_train,label_train)

#plot graph of feature importances for better visualization
    
feat_importances = pd.Series(model.feature_importances_, index=KacakTahmin.columns)
feat_importances.nlargest(SelectedParameterNum).plot(kind='barh')
plt.show()

SelectedParameters = feat_importances.nlargest(SelectedParameterNum) 
SelectedParametersLabels = SelectedParameters.index

# Parametreleri Ekrana Yazdırma

#for index in range(0,SelectedParameterNum):
#    print(SelectedParametersLabels[index])

# -------------
# Seçilen en iyi parametrelerden yeni bir dataframe oluşturma
    
FirstColumn = KacakTahminVerileri[SelectedParametersLabels[0]]
KacakTahminOpt = pd.Series.to_frame(FirstColumn)

for index in range(1,SelectedParameterNum): 
    KacakTahminOpt[SelectedParametersLabels[index]] = KacakTahmin[SelectedParametersLabels[index]]

# -------------
 # Test Kümesi Optimumu

from sklearn.cross_validation import train_test_split
XStd_Train, XStd_Test, y_train, y_test = train_test_split(KacakTahminOpt,KacakSonuc,test_size=0.33,random_state=0)

# ----------------------------------------------------------------------------

# 8.1 Random Forest Optimum

AlgorithmName = 'Random Forest Optimum Parameters (' + str(SelectedParameterNum) + ')'

from sklearn.ensemble import RandomForestClassifier
rfc2 = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state=250)

start = time()
rfc2.fit(XStd_Train,y_train)
end = time()
PrintRunTimeMS(AlgorithmName,' Fit',start,end)

start = time()
y_pred2 = rfc2.predict(XStd_Test)
end = time()
PrintRunTimeMS(AlgorithmName,' Predict',start,end)

ShowRMSE(AlgorithmName, y_test, y_pred2)
CalcConfMatrix(AlgorithmName, y_test, y_pred2, 1)
CalcKFold(AlgorithmName, rfc2, 10)

print(accuracy_score(y_test, y_pred2))


# ----------------------------------------------------------------------------
# 9. BPSO 

# Maksimum iterasyon : 100 (iters)
# Parçacık Sayısı : 50 (n_particles)
# Number of computations : 
# cognitive parameters c1 : 0,5
# social parameters c2 : 0,5
# intertia weight w : 0,9

# Initialize swarm, arbitrary

options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

dimensions = 165 # dimensions should be the number of features
#optimizer.reset()   
iters=50
n_particles=70
c1_value=options['c1']
c2_value=options['c2']
w=options['w']
k=options['k']
p=options['p']

optimizer = ps.discrete.BinaryPSO(n_particles, dimensions=dimensions, options=options)
# Perform optimization
#cost, pos = optimizer.optimize(f, print_step=100, iters=1000, verbose=2)
cost, pos = optimizer.optimize(f, iters, n_processes=None)

# -------------

# BPSO ile secilen feature lar ile tekrar çalıştırma

#sub_rfc = RandomForestClassifier(n_estimators=10, criterion = 'gini', random_state=250)
#classifier = LogisticRegression(random_state=250, max_iter=500)
#sub_rfc = LogisticRegression(random_state=250, max_iter=500)

classifier = RandomForestClassifier(n_estimators=9, criterion = 'entropy', random_state=250, max_depth=10, max_features='log2', min_samples_leaf=5, min_samples_split=2)
sub_rfc = RandomForestClassifier(n_estimators=9, criterion = 'entropy', random_state=250, max_depth=10, max_features='log2', min_samples_leaf=5, min_samples_split=2)


X_Selected_Features_Train = XStd_Train[:,pos==1]  # subset
X_Selected_Features_Test = XStd_Test[:,pos==1]  # subset

#Seçilen parametreleri bir dizide toplama işlemi

BPSOColumnIndexes = np.arange(dimensions)
BPSOColumnIndexesFull = np.zeros(dimensions)

i = 0
j = 0
while i < len(pos):
    if (pos[i]==1): 
        BPSOColumnIndexesFull[j] = i
        j += 1
    i += 1
    
BPSOColumnIndexesFull = BPSOColumnIndexesFull[BPSOColumnIndexesFull != 0] # 0 olan satırlar silinerek, sadece seçilen indexlerin dizide kalması sağlanıyor
BPSOSelectedColumns = getBPSOSelectedColumnNames()    # Seçilen kolonların başlıklaeı alınıyor

#DF_X_Selected_Features = pd.DataFrame(data = X_Selected_Features, index = range(X_Selected_Features.shape[0]), columns=KacakTahmin.iloc[:,RFEColumnIndexes].columns)


# Perform classification and store performance in P
#c0 = rfc
c0 = classifier.fit(XStd_Train, y_train.values.ravel())   # Tüm veriler ile tahmin
c1 = sub_rfc.fit(X_Selected_Features_Train, y_train.values.ravel())  # Seçilen alt veriler (sub features) ile tahmin

##
## # Compute performance
## main_performance = (c0.predict(XStd_Test) == y_test).mean()
##subset_performance = (c1.predict(X_Selected_Features2) == y_test).mean()
##

c0Pred = c0.predict(XStd_Test)
accuracy = accuracy_score(y_test, c0Pred)
accuracy = round(accuracy, 4)

print("BPSO Parametreler : " )
print("Maksimum İterasyon Sayısı (iters) = " + str(iters))
print("Parçacık Sayısı (n_particles) = " + str(n_particles))
print("c1 = " + str(c1_value) + " c2 = " + str(c2_value) + " w = " + str(w) + " k = " + str(k) + " p = " + str(p))
print("Main Accuracy: %.2f%%" % (accuracy * 100.0))

c1Pred = c1.predict(X_Selected_Features_Test)
sbaccuracy = accuracy_score(y_test, c1Pred)
sbaccuracy = round(sbaccuracy, 4)
print("Subset Accuracy: %.2f%%" % (sbaccuracy * 100.0))

##
## print('Main performance: %.3f' % (main_performance))
## print('Subset performance: %.3f' % (subset_performance))
##

AlgorithmName = 'BPSO (Optimum Parametre : ' + str(len(BPSOColumnIndexesFull)) + ')'
#ShowRMSE(AlgorithmName, y_test, c1Pred)
XStd_Train = X_Selected_Features_Train
CalcKFold(AlgorithmName, sub_rfc, 10)
CalcConfMatrix(AlgorithmName, y_test, c1Pred, 1)

BPSOColumnNames = getBPSOSelectedColumnNames()


# ----------------------------------------------------------------------------

# 10.Genetic Algoritms - https://pypi.org/project/sklearn-genetic/#description

'''
estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

selector = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="accuracy",
                              max_features=5,
                              n_population=50,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=40,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=10,
                              caching=True,
                              n_jobs=-1)
selector = selector.fit(XStd_Train,y_train.values.ravel())

# print(selector.support_)
'''
