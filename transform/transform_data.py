import glob
import pandas as pd
dataPath = './original/*.xlsx'
outPath = './out/'

def transform_data():

    for f in glob.glob(dataPath):
        f = f.replace("\\", '/')
        xls = pd.ExcelFile(f)
        sheet_names = xls.sheet_names
        for sn in sheet_names:
            df = xls.parse(sn)
            df["Hora"] = pd.to_timedelta(df["Hora"])
            df["timestamp"] = df["Fecha"] + df["Hora"]
            df.set_index("timestamp", inplace=True)
            del df["Hora"]
            del df["Fecha"]
            del df["Empresa"]
            del df["UNegocio"]
            print(df.info())
            time_range = pd.date_range(df.index[0], df.index[-1], freq='30min')
            df = df.reindex(time_range)
            df["Demanda"] = df["Demanda"].interpolate(method='linear')
            df.to_pickle(outPath + sn + '.pkl', compression='infer')


transform_data()