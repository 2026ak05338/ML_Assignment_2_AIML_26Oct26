from pathlib import Path
import joblib, pandas as pd, streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
ROOT=Path(__file__).resolve().parent
MODELS={'Logistic Regression':'logistic_regression.joblib','Decision Tree':'decision_tree.joblib','KNN':'knn.joblib','Naive Bayes':'naive_bayes.joblib','Support Vector Machine':'support_vector_machine.joblib'}
st.set_page_config(page_title='Bank Marketing Classifier Lab',page_icon='📊',layout='wide')
st.title('📊 Bank Marketing — classifier comparison')
st.write('Evaluate a saved classifier on labeled held-out CSV rows. This app does not fit or retrain models.')
upload=st.file_uploader('Upload test CSV (features plus target column `y`)',type='csv')
if upload is None:
    sample=ROOT/'test_data.csv'
    if not sample.exists(): st.warning('Run train.py first to create saved models and test data.'); st.stop()
    st.info('Using the included held-out test_data.csv. Upload another CSV to evaluate it instead.'); data=pd.read_csv(sample)
else: data=pd.read_csv(upload)
st.caption(f'Rows: {len(data):,} · Columns: {len(data.columns)}')
if 'y' not in data: st.error('CSV must include target column `y`.'); st.stop()
choice=st.selectbox('Classifier',list(MODELS)); path=ROOT/'model'/MODELS[choice]
if not path.exists(): st.error(f'Saved artifact missing: {path.name}. Run train.py and commit model files.'); st.stop()
model=joblib.load(path); X=data.drop(columns='y'); y=data.y.astype(str); expected=list(model.feature_names_in_)
missing=sorted(set(expected)-set(X.columns)); extra=sorted(set(X.columns)-set(expected))
if missing or extra: st.error(f'Feature columns mismatch. Missing: {missing or "none"}; extra: {extra or "none"}'); st.stop()
X=X[expected]; pred=model.predict(X); classes=list(model.named_steps['classifier'].classes_)
if set(y.unique())-set(classes): st.error(f'Target labels must be among: {classes}'); st.stop()
pos='yes' if 'yes' in classes else classes[-1]; prob=model.predict_proba(X)[:,classes.index(pos)]
auc=roc_auc_score((y==pos).astype(int),prob) if y.nunique()==2 else float('nan')
metrics={'Accuracy':accuracy_score(y,pred),f'AUC (positive={pos})':auc,'Precision':precision_score(y,pred,pos_label=pos,zero_division=0),'Recall':recall_score(y,pred,pos_label=pos,zero_division=0),'F1':f1_score(y,pred,pos_label=pos,zero_division=0),'MCC':matthews_corrcoef(y,pred)}
cols=st.columns(len(metrics))
for col,(label,value) in zip(cols,metrics.items()): col.metric(label,f'{value:.4f}')
st.subheader('Confusion matrix'); cm=confusion_matrix(y,pred,labels=classes)
st.dataframe(pd.DataFrame(cm,index=[f'Actual {c}' for c in classes],columns=[f'Predicted {c}' for c in classes]),use_container_width=True)
st.subheader('Classification report'); st.code(classification_report(y,pred,labels=classes,zero_division=0))
with st.expander('Preview test data'): st.dataframe(data.head(20),use_container_width=True)
