import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

import streamlit as st
import pip
pip.main(['install', 'seaborn'])

# visualization


# data analysis

warnings.filterwarnings("ignore")

st.title("Web Data Mining")
st.write("By: Muhammad Arbiansyafa Siswanto - 200411100195")
st.write("Grade: Penambangan Data C")
upload_data, preporcessing, modeling, implementation = st.tabs(
    ["Upload Data", "Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan pada percobaan ini adalah data Data Churn Telekomunikasi (Membersihkan Dataset Churn Pelanggan Orange Telecom)")
    st.write("link dataset :https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets?resource=download")
    st.write("Terdiri dari 270 dataset terdapat 13 atribut dan 2 kelas.")
    st.write("Heart Attack (Serangan Jantung) adalah kondisi medis darurat ketika darah yang menuju ke jantung terhambat.")

    uploaded_files = st.file_uploader(
        "Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df_train = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df_train)

with preporcessing:
    st.write("""# Preprocessing""")

    df_data_train = df_train.copy()
    df_data_test = df_train.copy()

    df_data_train.tail()

    df_data_train['International plan'].replace(
        ['No', 'Yes'], [0, 1], inplace=True)
    df_data_test['International plan'].replace(
        ['No', 'Yes'], [0, 1], inplace=True)
    df_data_train['International plan'].value_counts()

    df_data_train['Voice mail plan'].replace(
        ['No', 'Yes'], [0, 1], inplace=True)
    df_data_test['Voice mail plan'].replace(
        ['No', 'Yes'], [0, 1], inplace=True)
    df_data_train['Voice mail plan'].value_counts()

    st.write(df_data_train['Voice mail plan'].isnull().sum())
    st.write(df_data_train['International plan'].isnull().sum())

    st.write(df_data_train.head())

    df_data_train['Total charge amount'] = df_data_train['Total day charge'] + \
        df_data_train['Total eve charge'] + \
        df_data_train['Total night charge'] + \
        df_data_train['Total intl charge']
    df_data_train['Total call minutes'] = df_data_train['Total day minutes'] + \
        df_data_train['Total eve minutes'] + \
        df_data_train['Total night minutes'] + \
        df_data_train['Total intl minutes']
    df_data_train['Total number of calls'] = df_data_train['Total day calls'] + \
        df_data_train['Total eve calls'] + \
        df_data_train['Total night calls'] + df_data_train['Total intl calls']

    st.write(df_data_train.sample(10))

    df_data_test['Total charge amount'] = df_data_test['Total day charge'] + \
        df_data_test['Total eve charge'] + \
        df_data_test['Total night charge'] + df_data_test['Total intl charge']
    df_data_test['Total call minutes'] = df_data_test['Total day minutes'] + \
        df_data_test['Total eve minutes'] + \
        df_data_test['Total night minutes'] + \
        df_data_test['Total intl minutes']
    df_data_test['Total number of calls'] = df_data_test['Total day calls'] + \
        df_data_test['Total eve calls'] + \
        df_data_test['Total night calls'] + df_data_test['Total intl calls']

    st.write(df_data_test.sample(10))

with modeling:
    df_data_train = df_train.copy()
    df_data_test = df_train.copy()

    X_train = df_data_train.drop("Churn", axis=1)
    Y_train = df_data_train["Churn"]
    X_test = df_data_test.drop("Churn", axis=1)
    Y_test = df_data_test["Churn"]
    X_train.shape, Y_train.shape, X_test.shape

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)

    # Calculating Details
    acc_knn_train = round(knn.score(X_train, Y_train) * 100, 2)
    acc_knn_test = round(knn.score(X_test, Y_test) * 100, 2)
    print('KNN Train Score is : ',  acc_knn_train)
    print('KNN test Score is : ', acc_knn_test)
    # Calculating Prediction
    accuracy = accuracy_score(Y_test, Y_pred)

    print('Accuracy Score is  = ', accuracy)

    conf = confusion_matrix(Y_test, Y_pred)
    print('confusion matrix \n',  conf)

    sns.heatmap(conf, center=True)
    plt.show()
