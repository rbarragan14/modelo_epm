# SP: Cargar los paquetes necesarios/Pt: Carregar os pacotes necessários

import sys, os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import base64
import pickle

import warnings
warnings.filterwarnings('ignore')

# SP: Lista de columnas/Pt: Lista de colunas

colNames = ['Id', 'NR_TLFN','ID_LNHA','NR_CPF','NR_CPF_NUM','DS_CRCT_PLNO','ID_SIST_PGTO','MAX_DUSO_API05','MAX_DUSO_API41',
            'QTD_CNTTOS_POS_ENTE','QTD_CNTTOS_POS_SNTE','QTD_MIN_M1','QTD_MIN_ENTE_MM_M1','MAX_DUSO_API13',
            'MAX_DUSO_API01','QTD_CHMD_ENTE_ON_6M','QTD_API_DIST','MAX_DUSO_SEG13','MAX_DUSO_SEG11','MAX_DUSO_API70',
            'QTD_MIN_OF_6M','QTD_CHMD_ENTE_MM_6M','QTD_MIN_CONECT_6M','QTD_MB_6M','QTD_MIN_ENTE_MF_6M','SGMT_VZNC',
            'TROCA_MAIS_ANTIGA_DOIS_ANOS','TROCA_MAIS_ANTIGA_MAIS_TRES_ANOS','TROCA_MAIS_ANTIGA_SEM_TROCA',
            'TROCA_MAIS_ANTIGA_TRES_ANOS','TROCA_MAIS_ANTIGA_UM_ANO','SEGMENTACAO_FINAL_GOLD','SEGMENTACAO_FINAL_PLATINUM',
            'SEGMENTACAO_FINAL_V','TMP_LNHA_MENOR_IGUAL_50','TMP_PLNO_MENOR_IGUAL_100','VL_PLNO_MENOR_IGUAL_200','VL_TOTAL_MENOR_IGUAL_200']

# SP: Tipos de columnas/PT: Tipos de colunas

colTypes = dict([('Id', 'int64'),('NR_TLFN', 'int64'),('ID_LNHA', 'int64'),('NR_CPF', 'int64'),('NR_CPF_NUM', 'int64'),
                 ('DS_CRCT_PLNO', 'object'),('ID_SIST_PGTO', 'int64'),('MAX_DUSO_API05', 'int64'),
                 ('MAX_DUSO_API41', 'int64'),('QTD_CNTTOS_POS_ENTE', 'int64'),('QTD_CNTTOS_POS_SNTE', 'int64'),
                 ('QTD_MIN_M1', 'int64'),('QTD_MIN_ENTE_MM_M1', 'int64'),('MAX_DUSO_API13', 'int64'),
                 ('MAX_DUSO_API01', 'int64'),('QTD_CHMD_ENTE_ON_6M', 'int64'),('QTD_API_DIST', 'int64'),
                 ('MAX_DUSO_SEG13', 'int64'),('MAX_DUSO_SEG11', 'int64'),('MAX_DUSO_API70', 'int64'),
                 ('QTD_MIN_OF_6M', 'int64'),('QTD_CHMD_ENTE_MM_6M', 'int64'),('QTD_MIN_CONECT_6M', 'int64'),
                 ('QTD_MB_6M', 'int64'),('QTD_MIN_ENTE_MF_6M', 'int64'),('SGMT_VZNC', 'int64'),
                 ('TROCA_MAIS_ANTIGA_DOIS_ANOS', 'int64'),('TROCA_MAIS_ANTIGA_MAIS_TRES_ANOS', 'int64'),
                 ('TROCA_MAIS_ANTIGA_SEM_TROCA', 'int64'),('TROCA_MAIS_ANTIGA_TRES_ANOS', 'int64'),
                 ('TROCA_MAIS_ANTIGA_UM_ANO', 'int64'),('SEGMENTACAO_FINAL_GOLD', 'int64'),
                 ('SEGMENTACAO_FINAL_PLATINUM', 'int64'),('SEGMENTACAO_FINAL_V', 'int64'),
                 ('TMP_LNHA_MENOR_IGUAL_50', 'int64'),('TMP_PLNO_MENOR_IGUAL_100', 'int64'),
                 ('VL_PLNO_MENOR_IGUAL_200', 'int64'),('VL_TOTAL_MENOR_IGUAL_200', 'int64')])

# SP: Fuerza la cadena de entrada para que Python interprete los números correctamente/PT: Força a cadeia de entrada para que o Python interprete os números corretamente.

sciStrToInt = lambda x: int(float("".join(x.split())))

converters = {0 :sciStrToInt,1 :sciStrToInt,2 :sciStrToInt,3 :sciStrToInt,4 :sciStrToInt,6 :sciStrToInt,7 :sciStrToInt,
              8 :sciStrToInt,9 :sciStrToInt,10 :sciStrToInt,11 :sciStrToInt,12 :sciStrToInt,13 :sciStrToInt,
              14 :sciStrToInt,15 :sciStrToInt,16 :sciStrToInt,17 :sciStrToInt,18 :sciStrToInt,19 :sciStrToInt,
              20 :sciStrToInt,21 :sciStrToInt,22 :sciStrToInt,23 :sciStrToInt,24 :sciStrToInt,25 :sciStrToInt,
              26 :sciStrToInt,27 :sciStrToInt,28 :sciStrToInt,29 :sciStrToInt,30 :sciStrToInt,31 :sciStrToInt,
              32 :sciStrToInt,33 :sciStrToInt,34 :sciStrToInt,35 :sciStrToInt,36 :sciStrToInt,37 :sciStrToInt}

# Configuración del delimitador de Teradata
# Configuração do delimitador do Teradata

DELIMITER='\t'

# Carga y procesamiento de los registros
# Carga e processamento dos registros

df_sample = pd.read_csv(sys.stdin, sep=DELIMITER, header=None, names=colNames,
                        index_col=False, iterator=False, converters=converters)

# Para los AMP que no reciben datos, sale de la instancia de la secuencia de comandos con elegancia.
# Para AMPs que não recebem dados, sai da instância do script normalmente.

if df_sample.empty:
    sys.exit()

# Leer el archivo del modelo
# Ler o arquivo do modelo

model_t3 = pickle.load(open('./pocanalytics/model_gbc_alt_valor.pickle', 'rb'))
# model_t3 = pickle.load(open('./td01/model_gbc_alt_valor.pickle', 'rb'))

# Decodificar el modelo
# Decodificar o modelo

modelSer = base64.b64encode(pickle.dumps(model_t3)).decode('ascii')
model = pickle.loads(base64.b64decode(modelSer))

# Función de score
# Função de pontuação

prediction_ = model.predict_proba(df_sample.iloc[:,7:38])
score_ = pd.DataFrame(prediction_, columns=['Score_0', 'Score_1'])
df_sample['Score'] = score_['Score_1']

df_fin = df_sample[['Id','Score']]

#df_fin['Id'] = df_fin['Id'].astype('int64', errors='coerce')

df_fin['Score'] = df_fin['Score'].astype(float)

# Exportar resultados a la base de datos de Teradata Vantage a través de salida estándar
# Exportar resultados para o banco de dados Teradata Vantage por meio de saída padrão

for i in range(0, df_fin.shape[0]):
    print(df_fin.at[i,'Id'], DELIMITER, df_fin.at[i,'Score'])











