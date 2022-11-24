import streamlit as st
import pandas as pd
import numpy as np


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("DATAMINING")
st.write("MUHAMMAD ARBIANSYAFA SISWANTO - 200411100195")
st.write("Penambangan Data C")
upload_data, preporcessing, modeling, implementation = st.tabs(["Upload Data", "Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah Telecom Churn Dataset Cleaned Orange dataset ini diambil dari https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets")
    st.write("Total dataset yang dugunakan sebanyak 3335 data, dengan data training 80% berjumlah (2667) dan data testing 20% berjumlah (668)")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)


with preporcessing:
    st.write("""# Preprocessing""")
    df[["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "smoking_status"]].agg(['min','max'])

    df.stroke.value_counts()
    df = df.drop(columns=["id","bmi"])

    X = df.drop(columns="stroke")
    y = df.stroke
    "### Membuang fitur yang tidak diperlukan"
    df

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.stroke).columns.values.tolist()

    "### Label"
    labels

    st.markdown("# Normalize")

    "### Normalize data"

    dataubah=df.drop(columns=['gender','ever_married','work_type','Residence_type','smoking_status'])
    dataubah

    "### Normalize data gender"
    data_gen=df[['gender']]
    gen = pd.get_dummies(data_gen)
    gen

    # "### Normalize data Hypertension"
    # data_hypertension=df[['hypertension']]
    # hypertension = pd.get_dummies(data_hypertension)
    # hypertension

    "### Normalize data married"
    data_married=df[['ever_married']]
    married = pd.get_dummies(data_married)
    married

    "### Normalize data work"
    data_work=df[['work_type']]
    work = pd.get_dummies(data_work)
    work

    "### Normalize data residence"
    data_residence=df[['Residence_type']]
    residence = pd.get_dummies(data_residence)
    residence

    "### Normalize data smoke"
    data_smoke=df[['smoking_status']]
    smoke = pd.get_dummies(data_smoke)
    smoke

    dataOlah = pd.concat([gen,married,work,residence,smoke], axis=1)
    dataHasil = pd.concat([df,dataOlah], axis = 1)

    X = dataHasil.drop(columns=["gender","ever_married","work_type","Residence_type","smoking_status","stroke"])
    y = dataHasil.stroke
    "### Normalize data hasil"
    X

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(dataHasil.stroke).columns.values.tolist()
    
    "### Label"
    labels

    # """## Normalisasi MinMax Scaler"""


    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape



with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # X_train = cv.fit_transform(X_train)
    # X_test = cv.fit_transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

# with modeling:

#     st.markdown("# Model")
#     # membagi data menjadi data testing(20%) dan training(80%)
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#     # X_train.shape, X_test.shape, y_train.shape, y_test.shape

#     nb = st.checkbox("Metode Naive Bayes")
#     knn = st.checkbox("Metode KNN")
#     dt = st.checkbox("Metode Decision Tree")
#     sb = st.button("submit")

#     #Naive Bayes
#     # Feature Scaling to bring the variable in a single scale
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)

#     GaussianNB(priors=None)
#     # Fitting Naive Bayes Classification to the Training set with linear kernel
#     nvklasifikasi = GaussianNB()
#     nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

#     # Predicting the Test set results
#     y_pred = nvklasifikasi.predict(X_test)
        
#     y_compare = np.vstack((y_test,y_pred)).T
#     nvklasifikasi.predict_proba(X_test)

#     akurasi = round(100 * accuracy_score(y_test, y_pred))

#     #Decision tree
#     dt = DecisionTreeClassifier()
#     dt.fit(X_train, y_train)

#     # prediction
#     dt.score(X_test, y_test)
#     y_pred = dt.predict(X_test)
#     #Accuracy
#     akur = round(100 * accuracy_score(y_test,y_pred))

#     K=10
#     knn=KNeighborsClassifier(n_neighbors=K)
#     knn.fit(X_train,y_train)
#     y_pred=knn.predict(X_test)

#     skor_akurasi = round(100 * accuracy_score(y_test,y_pred))
    

#     if nb:
#         if sb:

#             """## Naive Bayes"""
            
#             st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))

#     if knn:
#         if sb:
#             """## KNN"""

#             st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    
#     if dt:
#         if sb:
#             """## Decision Tree"""
#             st.write('Model Decission Tree Accuracy Score: {0:0.2f}'.format(akur))

with implementation:
    st.write("# Implementation")
    Age = st.number_input('Masukkan Umur Pasien')

    # GENDER
    gender = st.radio("Gender",('Male', 'Female', 'Other'))
    if gender == "Male":
        gen_Female = 0
        gen_Male = 1
        gen_Other = 0
    elif gender == "Female" :
        gen_Female = 1
        gen_Male = 0
        gen_Other = 0
    elif gender == "Other" :
        gen_Female = 0
        gen_Male = 0
        gen_Other = 1

    # HYPERTENSION
    hypertension = st.radio("Hypertency",('No', 'Yes'))
    if hypertension == "Yes":
        hypertension = 1
    elif hypertension == "No":
        hypertension = 0
    
    # HEART
    heart_disease = st.radio("heart_disease",('No', 'Yes'))
    if heart_disease == "Yes":
        heart_disease = 1
        # heart_disease_N = 0
    elif heart_disease == "No":
        heart_disease = 0
        # heart_disease_N = 1

    # MARRIED
    ever_married = st.radio("ever_married",('No', 'Yes'))
    if ever_married == "Yes":
        ever_married_Y = 1
        ever_married_N = 0
    elif ever_married == "No":
        ever_married_Y = 0
        ever_married_N = 1

    # WORK
    work_type = st.radio("work_type",('Govt_job', 'Never_worked','Private', 'Self_employed', 'childern'))
    if work_type == "Govt_job":
        work_type_G = 1
        work_type_Never = 0
        work_type_P = 0
        work_type_S = 0
        work_type_C = 0
    elif work_type == "Never_worked":
        work_type_G = 0
        work_type_Never = 1
        work_type_P = 0
        work_type_S = 0
        work_type_C = 0
    elif work_type == "Private":
        work_type_G = 0
        work_type_Never = 0
        work_type_P = 1
        work_type_S = 0
        work_type_C = 0
    elif work_type == "Self_employed":
        work_type_G = 0
        work_type_Never = 0
        work_type_P = 0
        work_type_S = 1
        work_type_C = 0
    elif work_type == "childern":
        work_type_G = 0
        work_type_Never = 0
        work_type_P = 0
        work_type_S = 0
        work_type_C = 1

    # RESIDENCE
    residence_type = st.radio("residence_type",('Rural', 'Urban'))
    if residence_type == "Rural":
        residence_type_R = 1
        residence_type_U = 0
    elif residence_type == "Urban":
        residence_type_R = 0
        residence_type_U = 1

    # GLUCOSE
    avg_glucose_level = st.number_input('Masukkan Angka glukosa')
    
    # SMOKE
    smoking_status = st.radio("smoking_status",('Unknown', 'Formerly smoked', 'never smoked', 'smokes'))
    if smoking_status == "Unknown":
        smoking_status_U = 1
        smoking_status_F = 0
        smoking_status_N = 0
        smoking_status_S = 0
    elif smoking_status == "Formerly smoked":
        smoking_status_U = 0
        smoking_status_F = 1
        smoking_status_N = 0
        smoking_status_S = 0
    elif smoking_status == "never smoked":
        smoking_status_U = 0
        smoking_status_F = 0
        smoking_status_N = 1
        smoking_status_S = 0
    elif smoking_status == "smokes":
        smoking_status_U = 0
        smoking_status_F = 0
        smoking_status_N = 0
        smoking_status_S = 1
        
    bmi = st.number_input('Masukkan BMI')

    
    # Sex = st.radio(
    # "Masukkan Jenis Kelamin Anda",
    # ('Laki-laki','Perempuan'))
    # if Sex == "Laki-laki":
    #     Sex_Female = 0
    #     Sex_Male = 1
    # elif Sex == "Perempuan" :
    #     Sex_Female = 1
    #     Sex_Male = 0

    # BP = st.radio(
    # "Masukkan Tekanan Darah Anda",
    # ('Tinggi','Normal','Rendah'))
    # if BP == "Tinggi":
    #     BP_High = 1
    #     BP_LOW = 0
    #     BP_NORMAL = 0
    # elif BP == "Normal" :
    #     BP_High = 0
    #     BP_LOW = 0
    #     BP_NORMAL = 1
    # elif BP == "Rendah" :
    #     BP_High = 0
    #     BP_LOW = 1
    #     BP_NORMAL = 0

    # Cholesterol = st.radio(
    # "Masukkan Kadar Kolestrol Anda",
    # ('Tinggi','Normal'))
    # if Cholesterol == "Tinggi" :
    #     Cholestrol_High = 1
    #     Cholestrol_Normal = 0 
    # elif Cholesterol == "Normal":
    #     Cholestrol_High = 0
    #     Cholestrol_Normal = 1
        
    # Na_to_K = st.number_input('Masukkan Rasio Natrium Ke Kalium dalam Darah')



    def submit():
        # input
        inputs = np.array([[
            Age,
            hypertension,
            heart_disease,
            avg_glucose_level,
            gen_Female, gen_Male, gen_Other,
            ever_married_N, ever_married_Y,
            work_type_G, work_type_Never, work_type_P, work_type_S, work_type_C,
            residence_type_R, residence_type_U,
            smoking_status_U, smoking_status_F, smoking_status_N, smoking_status_S, bmi
            ]])
        # st.write(inputs)
        # baru = pd.DataFrame(inputs)
        # input = pd.get_dummies(baru)
        # st.write(input)
        # inputan = np.array(input)
        # import label encoder
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda dinyatakan : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

