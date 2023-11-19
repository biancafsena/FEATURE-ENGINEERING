{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "###Passo 1 - Instalar os módulos e importar"
      ],
      "metadata": {
        "id": "L2TQWJyzUbcW"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0OCErPuePz47"
      },
      "outputs": [],
      "source": [
        "pip install -U feature-engine"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "C_uHJr_EPz4-"
      },
      "outputs": [],
      "source": [
        "pip install category_encoders"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "M5qO99ctrfad"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import seaborn as sb\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, ConfusionMatrixDisplay\n",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.compose import ColumnTransformer\n",
        "\n",
        "\n",
        "from feature_engine.encoding import RareLabelEncoder, CountFrequencyEncoder\n",
        "import category_encoders as ce\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Processamento utilizando a coluna **\"artist\"**"
      ],
      "metadata": {
        "id": "R8aVkBCF3AMy"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0ZkAeC-xhd04"
      },
      "source": [
        "###Passo 2 - Carga do Dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZVaf_v1ohltE"
      },
      "outputs": [],
      "source": [
        "url = 'https://telescopeinstorage.blob.core.windows.net/datasets/DadosSpotify.csv'\n",
        "dataset = pd.read_csv(url, engine='python')\n",
        "dataset.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Grt1ru8dxPF1"
      },
      "outputs": [],
      "source": [
        "dataset.shape"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "l8THvs-8s5E0"
      },
      "source": [
        "###Passo 3 - Pré-processamento"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8YKSPSsSr0jh"
      },
      "outputs": [],
      "source": [
        "dataset.info()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "55Oce1vGjWZT"
      },
      "source": [
        "###Passo 4 - Separando as colunas"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "dECMogpEcYkQ"
      },
      "outputs": [],
      "source": [
        "Spotify = dataset[['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence','target','artist']]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "L9q99mRNeM-r"
      },
      "outputs": [],
      "source": [
        "Spotify.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dyB7eZzCoxrB"
      },
      "outputs": [],
      "source": [
        "Spotify.shape"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rFuIdU3Mh0Q3"
      },
      "source": [
        "###Passo 5 - Separação do Conjunto de Treinamento e Teste"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#####Excluíndo a coluna **\"Target**\""
      ],
      "metadata": {
        "id": "lR6eajmDkF-A"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "vfb6kDK7hW_V"
      },
      "outputs": [],
      "source": [
        "X = Spotify.drop('target', axis=1)\n",
        "y = Spotify['target']"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uw99xgwAh3fq"
      },
      "source": [
        "###Passo 6 - Dividindo treino e teste (teste com 30%)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "AwtnNpP1fAYP"
      },
      "outputs": [],
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_train.shape, X_test.shape"
      ],
      "metadata": {
        "id": "VYsIc9V3qWmY",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6c1c8b48-5ff4-4609-b380-fc9e15af39ea"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((1411, 14), (606, 14))"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uindSZvDPz5E"
      },
      "source": [
        "#Para a execução somente do CountFrequency siga o **\"Passo 7 até 10\"**\n",
        "#Para a execução somente do OneHotEncoder siga para o **\"Passo 11**\""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "m8vaFTGLi2t6"
      },
      "source": [
        "###Passo 7 - Processamento da coluna \"artist\" Count Frequency"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zl5gqdR2fWKU"
      },
      "outputs": [],
      "source": [
        "encoder = CountFrequencyEncoder(encoding_method='frequency', variables=['artist'])\n",
        "\n",
        "encoder.fit(X_train)\n",
        "\n",
        "X_train = encoder.transform(X_train)\n",
        "X_test = encoder.transform(X_test)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "v45LiH3kl-7v"
      },
      "source": [
        "#No processo utilizando o Count Frequency Encoder, acabou gerando valores nulos no dataset"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dJU8gleCoBrW"
      },
      "source": [
        "###Passo 8 - X_train - Valores corretos"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "896JmgIpj0P9"
      },
      "outputs": [],
      "source": [
        "X_train.isnull().sum()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "N91nbWopmmBW"
      },
      "source": [
        "###Passo 9 - Validando valores nulos, erros encontrados na execução **\"Processo de teste com 30% dos dados\"**\n",
        "\n",
        "#####X_test apresentou 350 valores nulos na coluna \"Artist\""
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_test.isnull().sum()"
      ],
      "metadata": {
        "id": "w5FsOW36lOH5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gSnh8AVCmHyh"
      },
      "source": [
        "###Passo 10- Correção\n",
        "#####Inseridos valores 0 para os campos nulos"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_test['artist'].fillna(0, inplace=True)"
      ],
      "metadata": {
        "id": "g-FJsgQ3lSWb"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0DSj4hrcxs2y"
      },
      "outputs": [],
      "source": [
        "X_test.isnull().sum()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Y44sCeaWqkYc"
      },
      "outputs": [],
      "source": [
        "X_train.info()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "V9Gs1T4vPz5M"
      },
      "source": [
        "#Passo 11 - Processamento da coluna \"artist\" OneHotEncoder"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')\n",
        "X_train = encoder.fit_transform(X_train[['artist']])\n",
        "X_test = encoder.transform(X_test[['artist']])"
      ],
      "metadata": {
        "id": "YMaJ4hTytm3z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kN9S9Lscoad4"
      },
      "source": [
        "###Passo 12 - Normalização e Scaling do Conjunto"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "Mj12TxUuPz5M"
      },
      "outputs": [],
      "source": [
        "scaler = StandardScaler().fit(X_train)\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.transform(X_test)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Passo 13 - Verificar o conjunto"
      ],
      "metadata": {
        "id": "55RApKuwVTuj"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0DqJxELU1GPu"
      },
      "outputs": [],
      "source": [
        "X_train"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "j_3AO4ln1G4h"
      },
      "outputs": [],
      "source": [
        "X_test"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "K9o-3eIzPz5M"
      },
      "source": [
        "###Passo 14 - Combinando atributos numéricos e codificados"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "id": "DnNKd2K_Pz5M"
      },
      "outputs": [],
      "source": [
        "X_train = np.hstack((X_train, X_train))\n",
        "X_test = np.hstack((X_test, X_test))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "n_U7gvS_tA7y"
      },
      "source": [
        "###Passo  15 - Classificação"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eRaI244VolCn"
      },
      "source": [
        "#Modelo de Classificador - Remover o \"#\" para a execução"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gwze6B-2o1pb"
      },
      "source": [
        "#Regressão Logística"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "tbzp6bARorE8"
      },
      "outputs": [],
      "source": [
        "model = LogisticRegression()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SwsbDklkpR8b"
      },
      "outputs": [],
      "source": [
        "model.fit(X_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "id": "pD3EInOpqSYB"
      },
      "outputs": [],
      "source": [
        "y_pred_rl = model.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "id": "J60l7MtiPz5N",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4f5a50ae-15eb-4974-a758-33b55b04218c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Acurácia do modelo: 64.19%\n"
          ]
        }
      ],
      "source": [
        "accuracy = accuracy_score(y_test, y_pred_rl)\n",
        "print(\"Acurácia do modelo: {:.2f}%\".format(accuracy * 100))"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm = confusion_matrix(y_test, y_pred_rl)\n",
        "disp = ConfusionMatrixDisplay(cm)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "o83BpFb7f0WN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "labels = [0,1]\n",
        "label_names = ['NS', 'S']\n",
        "cm = confusion_matrix(y_test, y_pred_rl, labels = labels, normalize='true')\n",
        "disp = ConfusionMatrixDisplay(cm, display_labels = label_names)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "9u5yXlWUfvEn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jxmTYV0iozrN"
      },
      "source": [
        "#scikit-learn"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 34,
      "metadata": {
        "id": "jJfz602RhP4X"
      },
      "outputs": [],
      "source": [
        "classifier = DecisionTreeClassifier()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IpCB63TwpVXi"
      },
      "outputs": [],
      "source": [
        "classifier.fit(X_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 36,
      "metadata": {
        "id": "mjeea0YaqUvN"
      },
      "outputs": [],
      "source": [
        "y_pred_sl = classifier.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CcqMeSy4Pz5N"
      },
      "outputs": [],
      "source": [
        "print(classification_report(y_test, y_pred_sl))\n",
        "print(\"Acurácia do modelo: {:.2f}%\".format(accuracy * 100))"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cm = confusion_matrix(y_test, y_pred_sl)\n",
        "disp = ConfusionMatrixDisplay(cm)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "VcW2LMiqfeof"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "labels = [0,1]\n",
        "label_names = ['NS', 'S']\n",
        "cm = confusion_matrix( y_test, y_pred_sl, labels = labels, normalize='true')\n",
        "disp = ConfusionMatrixDisplay(cm, display_labels = label_names)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "tcvKjm2FfhRN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YsK3qBKlow1b"
      },
      "source": [
        "#Criação do modelo **KNN**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 41,
      "metadata": {
        "id": "4HmV1bzIPz5O"
      },
      "outputs": [],
      "source": [
        "Classif_KNN = KNeighborsClassifier(n_neighbors=1)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3h_cDIfjtmkq"
      },
      "outputs": [],
      "source": [
        "Classif_KNN.fit(X_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 43,
      "metadata": {
        "id": "SdTGhaxvt9IM"
      },
      "outputs": [],
      "source": [
        "y_pred_knn = Classif_KNN.predict(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(classification_report(y_test, y_pred_knn))\n",
        "accuracy_knn = accuracy_score(y_test, y_pred_knn)\n",
        "print(f'Acurácia do KNN: {accuracy_knn}')"
      ],
      "metadata": {
        "id": "B19Ay8Tka81F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cm = confusion_matrix(y_test, y_pred_knn)\n",
        "disp = ConfusionMatrixDisplay(cm)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "_PuoTvklXMsV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "labels = [0,1]\n",
        "label_names = ['NS', 'S']\n",
        "cm = confusion_matrix( y_test, y_pred_knn, labels = labels, normalize='true')\n",
        "disp = ConfusionMatrixDisplay(cm, display_labels = label_names)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "Ujn7ctcfXNXu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Teste - **KNN** do K=1 até K=20"
      ],
      "metadata": {
        "id": "79_yhyUcX2JX"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 48,
      "metadata": {
        "id": "34ukQ8aG2pjo"
      },
      "outputs": [],
      "source": [
        "k_range = range(1, 20)\n",
        "scores = []\n",
        "\n",
        "for k in k_range:\n",
        "    knn = KNeighborsClassifier(n_neighbors=k)\n",
        "    knn.fit(X_train, y_train)\n",
        "    y_pred_knn = knn.predict(X_test)\n",
        "    scores.append(accuracy_score(y_test, y_pred_knn))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tzqQEjyM2rZf"
      },
      "outputs": [],
      "source": [
        "print(scores)\n",
        "#Plota os valores de acc. em função do valor escolhido de K\n",
        "plt.plot(k_range, scores)\n",
        "plt.xlabel('Value of K for KNN')\n",
        "plt.ylabel('Testing Accuracy')"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Avaliando a melhor configuração de **KNN**"
      ],
      "metadata": {
        "id": "xqihHqWlXl4u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "best_KNN = KNeighborsClassifier(n_neighbors=5)\n",
        "best_KNN.fit(X_train, y_train)\n",
        "y_pred_knn2 = best_KNN.predict(X_test)"
      ],
      "metadata": {
        "id": "V6tAUfouXnPE"
      },
      "execution_count": 50,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(classification_report(y_test, y_pred_knn2))\n",
        "accuracy_knn2 = accuracy_score(y_test, y_pred_knn2)\n",
        "print(f'Acurácia do KNN: {accuracy_knn2}')"
      ],
      "metadata": {
        "id": "KXEeTLqRXnFr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "N7u1Z4uwyBtj"
      },
      "outputs": [],
      "source": [
        "cm = confusion_matrix(y_test, y_pred_knn2)\n",
        "disp = ConfusionMatrixDisplay(cm)\n",
        "disp.plot()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QB4hIkJ_zsIs"
      },
      "outputs": [],
      "source": [
        "labels = [0,1]\n",
        "label_names = ['NS', 'S']\n",
        "cm = confusion_matrix( y_test, y_pred_knn2, labels = labels, normalize='true')\n",
        "disp = ConfusionMatrixDisplay(cm, display_labels = label_names)\n",
        "disp.plot()"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Processo sem a coluna **\"artist\"**"
      ],
      "metadata": {
        "id": "Kt_Ig0nj3N5z"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Passo 2 - Carga do Dataset"
      ],
      "metadata": {
        "id": "GlICCOus3gYs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "url = 'https://telescopeinstorage.blob.core.windows.net/datasets/DadosSpotify.csv'\n",
        "dataset = pd.read_csv(url, engine='python')\n",
        "dataset.head()"
      ],
      "metadata": {
        "id": "W_3whKjc3WwR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Passo 3 - Separando as colunas"
      ],
      "metadata": {
        "id": "y_JdaQCJ3nrj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "Spotify = dataset[['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence','target','artist']]"
      ],
      "metadata": {
        "id": "eU1bUnLf3cA-"
      },
      "execution_count": 56,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Passo 4 - Separação do Conjunto de Treinamento e Teste"
      ],
      "metadata": {
        "id": "oRW0Yt0b3q2R"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X = Spotify.drop(['target', 'artist'], axis=1)\n",
        "y = Spotify['target']"
      ],
      "metadata": {
        "id": "7IHVOA4E3wEV"
      },
      "execution_count": 58,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Passo 5 - Dividindo treino e teste (teste com 30%)"
      ],
      "metadata": {
        "id": "OydoNhXD3wfT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)"
      ],
      "metadata": {
        "id": "wVHG2LgS4RJF"
      },
      "execution_count": 59,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train.shape, X_test.shape"
      ],
      "metadata": {
        "id": "_MCLTFCS4WYM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Passo 6 - Normalização e Scaling do Conjunto"
      ],
      "metadata": {
        "id": "lYeMNLjb4W9c"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "scaler = StandardScaler().fit(X_train)\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.transform(X_test)"
      ],
      "metadata": {
        "id": "sd4mIMFM4XZf"
      },
      "execution_count": 61,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Passo 7 - Combinando atributos numéricos e codificados"
      ],
      "metadata": {
        "id": "_tKMUqcn4kJd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = np.hstack((X_train, X_train))\n",
        "X_test = np.hstack((X_test, X_test))"
      ],
      "metadata": {
        "id": "60mysMRK4kv9"
      },
      "execution_count": 62,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Regressão Logística"
      ],
      "metadata": {
        "id": "mnrYF4Sa4qmh"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = LogisticRegression()"
      ],
      "metadata": {
        "id": "xbV6oN8e4q50"
      },
      "execution_count": 63,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(X_train, y_train)"
      ],
      "metadata": {
        "id": "T2Tqst6E41_9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred_rl = model.predict(X_test)"
      ],
      "metadata": {
        "id": "xTxKuD_Y46Lt"
      },
      "execution_count": 65,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accuracy = accuracy_score(y_test, y_pred_rl)\n",
        "print(\"Acurácia do modelo: {:.2f}%\".format(accuracy * 100))"
      ],
      "metadata": {
        "id": "8IooVq8V49kZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Criação do modelo KNN"
      ],
      "metadata": {
        "id": "ekDw3gBm5Af-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "Classif_KNN = KNeighborsClassifier(n_neighbors=1)"
      ],
      "metadata": {
        "id": "5-BtzPHV5Ed5"
      },
      "execution_count": 67,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Classif_KNN.fit(X_train, y_train)"
      ],
      "metadata": {
        "id": "--F__NwG5HfW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred_knn = Classif_KNN.predict(X_test)"
      ],
      "metadata": {
        "id": "oZ_0Hs-F5HaX"
      },
      "execution_count": 69,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(classification_report(y_test, y_pred_knn))\n",
        "accuracy_knn = accuracy_score(y_test, y_pred_knn)\n",
        "print(f'Acurácia do KNN: {accuracy_knn}')"
      ],
      "metadata": {
        "id": "lSvc2fkw5HQH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cm = confusion_matrix(y_test, y_pred_knn)\n",
        "disp = ConfusionMatrixDisplay(cm)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "U2uuiXlm5P_k"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "labels = [0,1]\n",
        "label_names = ['NS', 'S']\n",
        "cm = confusion_matrix( y_test, y_pred_knn, labels = labels, normalize='true')\n",
        "disp = ConfusionMatrixDisplay(cm, display_labels = label_names)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "sqxymnos5TtX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Teste - KNN do K=1 até K=20"
      ],
      "metadata": {
        "id": "IMeDZxwV5VHx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "k_range = range(1, 20)\n",
        "scores = []\n",
        "\n",
        "for k in k_range:\n",
        "    knn = KNeighborsClassifier(n_neighbors=k)\n",
        "    knn.fit(X_train, y_train)\n",
        "    y_pred_knn = knn.predict(X_test)\n",
        "    scores.append(accuracy_score(y_test, y_pred_knn))"
      ],
      "metadata": {
        "id": "oQkg-Cvs5WNZ"
      },
      "execution_count": 73,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(scores)\n",
        "#Plota os valores de acc. em função do valor escolhido de K\n",
        "plt.plot(k_range, scores)\n",
        "plt.xlabel('Value of K for KNN')\n",
        "plt.ylabel('Testing Accuracy')"
      ],
      "metadata": {
        "id": "xsNzNEeh5XcV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Avaliando a melhor configuração de **KNN**"
      ],
      "metadata": {
        "id": "gHbm3Tiy5ibb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "best_KNN = KNeighborsClassifier(n_neighbors=5)\n",
        "best_KNN.fit(X_train, y_train)\n",
        "y_pred_knn2 = best_KNN.predict(X_test)"
      ],
      "metadata": {
        "id": "N1iTA2FP5Wlr"
      },
      "execution_count": 75,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(classification_report(y_test, y_pred_knn2))\n",
        "accuracy_knn2 = accuracy_score(y_test, y_pred_knn2)\n",
        "print(f'Acurácia do KNN: {accuracy_knn2}')"
      ],
      "metadata": {
        "id": "QNWoJej_5l4n"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cm = confusion_matrix(y_test, y_pred_knn2)\n",
        "disp = ConfusionMatrixDisplay(cm)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "NxEaD3Ss5p9o"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "labels = [0,1]\n",
        "label_names = ['NS', 'S']\n",
        "cm = confusion_matrix( y_test, y_pred_knn2, labels = labels, normalize='true')\n",
        "disp = ConfusionMatrixDisplay(cm, display_labels = label_names)\n",
        "disp.plot()"
      ],
      "metadata": {
        "id": "Ql5q9JNm5rCp"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.6"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}