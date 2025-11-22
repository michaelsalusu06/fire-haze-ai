import pandas as pd

def color_from_risk(risk: pd.Series) -> list[str]:
    palette = {0:"#00b050",1:"#66c266",2:"#ffd24d",3:"#ffb84d",4:"#ff704d",5:"#ff3333"}
    return [palette.get(int(x), "#aaaaaa") for x in risk.fillna(0)]
