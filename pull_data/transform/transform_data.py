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
            print("\n --> " + sn)
            df = xls.parse(sn)
            df["Hora"] = pd.to_timedelta(df["Hora"])
            df["timestamp"] = df["Fecha"] + df["Hora"]
            df.set_index("timestamp", inplace=True)
            del df["Hora"]
            del df["Fecha"]
            del df["Empresa"]
            del df["UNegocio"]
            date_ini = "01/01/2014"
            time_range = pd.date_range(date_ini, df.index[-1], freq='30min')
            df = df.reindex(time_range)
            df["Demanda"] = df["Demanda"].interpolate(method='linear')
            df.dropna(inplace=True)
            print(df.info())
            df.to_pickle(outPath + sn + '.pkl', compression='infer')


transform_data()