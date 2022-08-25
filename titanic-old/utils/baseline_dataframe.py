import numpy as  np 
import pandas as pd 
from .transformers import CabinExtraction, FamilyPresence

class Relevant_Cols:
    
    def __init__(self, 
                 path, 
                 use_cols, 
                 combine_family=True, 
                 extract_Cabin_Initial=True):
        self.path = path
        self.use_cols = use_cols
        self.combine_family = combine_family
        self.extract_Cabin_Initial = extract_Cabin_Initial
        
    def modified_dataframe(self):
        df = pd.read_csv(self.path, usecols=self.use_cols, index_col="PassengerId")
        
        if self.extract_Cabin_Initial:
            cabin_ex = CabinExtraction()
            transformer1 = cabin_ex.fit(df[['Cabin']])
            transformer1.transform(df[['Cabin']])
            df["Cabin"] = transformer1.transform(df[['Cabin']])
            
        if self.combine_family:
            famil_pres = FamilyPresence()
            transformer2 = famil_pres.fit(df)
            df["HasFamily"] = transformer2.transform(df)
            df.drop(["SibSp","Parch"], axis="columns", inplace=True)
        
        return df
