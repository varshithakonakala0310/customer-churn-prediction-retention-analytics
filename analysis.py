import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
BASE=os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(BASE,"data"),exist_ok=True); os.makedirs(os.path.join(BASE,"results"),exist_ok=True)
rng=np.random.default_rng(42); n=3000
tenure=rng.integers(1,73,n); monthly=rng.normal(65,25,n).clip(20,120); support=rng.integers(0,6,n); contract=rng.choice([0,1,2],n,p=[0.55,0.25,0.20])
logit=-1.2 -0.04*tenure +0.01*monthly +0.45*support -0.9*(contract==2) -0.3*(contract==1)
prob=1/(1+np.exp(-logit)); churn=rng.binomial(1,prob)
df=pd.DataFrame({"customer_id":range(1,n+1),"tenure_months":tenure,"monthly_charges_gbp":monthly.round(2),"support_tickets":support,"contract_type":contract,"internet_service":rng.choice(["DSL","Fibre","None"],n),"churn":churn})
df.to_csv(os.path.join(BASE,"data","churn_data_synthetic.csv"),index=False)
X=pd.get_dummies(df.drop(columns=["customer_id","churn"]),drop_first=True); y=df["churn"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
m=RandomForestClassifier(n_estimators=300,random_state=42); m.fit(Xtr,ytr); pred=m.predict(Xte); proba=m.predict_proba(Xte)[:,1]
acc=accuracy_score(yte,pred); auc=roc_auc_score(yte,proba)
imp=sorted(zip(X.columns,m.feature_importances_),key=lambda x:-x[1])
with open(os.path.join(BASE,"results","results_summary.txt"),"w") as f:
    f.write("Customer Churn Prediction — SYNTHETIC RESULTS\n")
    f.write(f"Customers: {n}, Churn rate: {y.mean():.4f}\n")
    f.write(f"Random Forest — accuracy={acc:.4f} ROC-AUC={auc:.4f} precision={precision_score(yte,pred):.4f} recall={recall_score(yte,pred):.4f} f1={f1_score(yte,pred):.4f}\n")
    f.write("Feature importance: " + ", ".join(f"{k}={v:.3f}" for k,v in imp) + "\n")
print(f"acc={acc:.4f} auc={auc:.4f} churn={y.mean():.4f}")
