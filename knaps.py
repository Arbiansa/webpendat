import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.write(""" 
# Cek data
""")

st.write("=========================================================================")

tab1, tab2, tab3, tab4 = st.tabs(["Import Data", "Preprocessing", "Modelling", "Evalutions"])

with tab1:
    st.write("Import Data")
    data = pd.read_csv("https://raw.githubusercontent.com/Feb11F/dataset/main/credit_score.csv")
    st.dataframe(data)

with tab2:
    data.head()

    X = data.drop(columns=["risk_rating"])

    X.head()

    # Mengambil kolom Rata-rata overdue dan mentranformasi menggunakan one-hot encoding
    rata_overdue = pd.get_dummies(X["rata_rata_overdue"], prefix="overdue")
    X = X.join(rata_overdue)

    X = X.drop(columns="rata_rata_overdue")

    labels = data["risk_rating"]
    # 
    KPR_status = pd.get_dummies(X["kpr_aktif"], prefix="KPR")
    X = X.join(KPR_status)

    # remove "rata_rata_overdue" feature
    X = X.drop(columns = "kpr_aktif")

    st.write("Menampilkan dataframe yang rata-rata overdue, risk rating dan kpr aktif sudah di drop")
    st.dataframe(X)

    st.write(" ## Normalisasi")
    st.write("Normalize feature 'pendapatan_setahun_juta', 'durasi_pinjaman_bulan', 'jumlah_tanggungan'")
    old_normalize_feature_labels = ['pendapatan_setahun_juta', 'durasi_pinjaman_bulan', 'jumlah_tanggungan']
    new_normalized_feature_labels = ['norm_pendapatan_setahun_juta', 'norm_durasi_pinjaman_bulan', 'norm_jumlah_tanggungan']
    normalize_feature = data[old_normalize_feature_labels]

    st.dataframe(normalize_feature)

    scaler = MinMaxScaler()

    scaler.fit(normalize_feature)

    normalized_feature = scaler.transform(normalize_feature)

    normalized_feature_df = pd.DataFrame(normalized_feature, columns = new_normalized_feature_labels)

    st.write("Data setelah dinormalisasi")
    st.dataframe(normalized_feature_df)

    X = X.drop(columns = old_normalize_feature_labels)

    X = X.join(normalized_feature_df)

    X = X.join(labels)

    st.write("Dataframe X baru")
    st.dataframe(X)

    subject_lables = ["Unnamed: 0",  "kode_kontrak"]
    X = X.drop(columns = subject_lables)

    # percent_amount_of_test_data = / HUNDRED_PERCENT
    percent_amount_of_test_data = 0.3

    st.write("Dataframe X baru yang tidak ada fitur/kolom unnamed: 0 dan kode kontrak")
    st.dataframe(X)
    st.write("## Hitung Data")
    st.write("- Pisahkan kolom risk rating dari data frame")
    st.write("- Ambil kolom 'risk rating' sebagai target kolom untuk kategori kelas")
    st.write("- Pisahkan data latih dengan data tes")
    st.write("""            Spliting Data

                data latih (nilai data)
                X_train 

                data tes (nilai data)
                X_test 

                data latih (kelas data)
                y_train

                data tes (kelas data)
                y_test""")

    # separate target 

    # values
    matrices_X = X.iloc[:,0:10].values

    # classes
    matrices_Y = X.iloc[:,10].values

    X_1 = X.iloc[:,0:10].values
    Y_1 = X.iloc[:, -1].values

    # X_train, X_test, y_train, y_test = train_test_split(matrices_X, matrices_Y, test_size = percent_amount_of_test_data, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, test_size = percent_amount_of_test_data, random_state=0)

    st.write("Menampilkan Y_1")
    st.write(Y_1)
    
    st.write("Menampilkan X_1")
    st.write(X_1)
    ### Dictionary to store model and its accuracy

    model_accuracy = OrderedDict()

    ### Dictionary to store model and its precision

    model_precision = OrderedDict()

    ### Dictionary to store model and its recall

    model_recall = OrderedDict()