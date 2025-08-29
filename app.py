# --- 1. Imports and Page Configuration ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from datetime import datetime

st.set_page_config(layout="wide", page_title="Organizational Health Intelligence Platform")

# --- 2. Professional Styling (CSS) ---
st.markdown("""
<style>
/* Main Background */
.stApp {
    background-color: #F0F2F6;
}
/* Main titles */
h1 {
    color: #004080; /* Darker Blue */
    font-family: 'Arial Black', Gadget, sans-serif;
}
h2, h3, h4 {
    color: #005A9C; /* Primary Blue */
    font-family: 'Arial', sans-serif;
}

/* Expander styling */
.st-emotion-cache-1r6slb0 { /* Expander Header */
    background-color: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #005A9C;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

/* Metric styling */
.st-emotion-cache-1f1figz, .st-emotion-cache-1b0udgb { /* Metric value and label */
    font-family: 'Arial', sans-serif;
}
.st-emotion-cache-1f1figz .st-emotion-cache-1g6goon { /* The big number in the metric */
    font-size: 2.5rem !important;
    font-weight: bold !important;
    color: #004080;
}

/* Custom Alert Boxes */
.stAlert > div {
    border-radius: 10px;
    border-width: 2px;
}

/* Styling for Input Widgets */
[data-testid="stSelectbox"] > div, [data-testid="stTextInput"] > div, [data-testid="stNumberInput"] > div {
    border: 2px solid #005A9C;
    border-radius: 10px;
    background-color: #FFFFFF;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
[data-testid="stSelectbox"] > div:hover, [data-testid="stTextInput"] > div:focus-within, [data-testid="stNumberInput"] > div:focus-within {
    border-color: #004080;
}
</style>
""", unsafe_allow_html=True)

# --- 3. Core Application Logic (including previous work and notebook logic) ---
#<editor-fold desc="Helper Functions & Config">
DISEASE_MODELS_INFO = {
    "HeartAttack": {"display_name": "Heart Attack", "importance_file": 'heart_attack_feature_importances_v12.joblib', "ml_score_col": "HeartAttack_RiskScore", "ml_zone_col": "HeartAttack_MLRiskZone", "rule_score_col": "HeartAttack_RiskScore_Calculated", "validation_col": "HeartAttack_Validation", "justification_col": "HeartAttack_Justification", "target_col": "HeartAttack_Target", "pred_col": "HeartAttack_ML_Prediction", "enabled": True, "key_params": ['LDL_CHOLESTEROL', 'CHOLESTEROL', 'BPSys', 'SMOKER', 'BMI']},
    "Diabetes": {"display_name": "Diabetes", "importance_file": 'diabetes_feature_importances_v12.joblib', "ml_score_col": "Diabetes_RiskScore", "ml_zone_col": "Diabetes_MLRiskZone", "rule_score_col": "Diabetes_RiskScore_Calculated", "validation_col": "Diabetes_Validation", "justification_col": "Diabetes_Justification", "target_col": "Diabetes_Target", "pred_col": "Diabetes_ML_Prediction", "enabled": True, "key_params": ['HbA1C', 'FBS', 'BMI', 'TRIGLYCERIDE']},
    "KidneyDisease": {"display_name": "Kidney Disease", "importance_file": 'kidney_disease_feature_importances_v12.joblib', "ml_score_col": "KidneyDisease_RiskScore", "ml_zone_col": "KidneyDisease_MLRiskZone", "rule_score_col": "KidneyDisease_RiskScore_Calculated", "validation_col": "KidneyDisease_Validation", "justification_col": "KidneyDisease_Justification", "target_col": "KidneyDisease_Target", "pred_col": "KidneyDisease_ML_Prediction", "enabled": True, "key_params": ['CREATININE', 'ALBUMIN_Serum_Categorical', 'Hb_HEMOGLOBIN']},
    "LiverDisease": {"display_name": "Liver Disease", "importance_file": 'liver_disease_feature_importances_v12.joblib', "ml_score_col": "LiverDisease_RiskScore", "ml_zone_col": "LiverDisease_MLRiskZone", "rule_score_col": "LiverDisease_RiskScore_Calculated", "validation_col": "LiverDisease_Validation", "justification_col": "LiverDisease_Justification", "target_col": "LiverDisease_Target", "pred_col": "LiverDisease_ML_Prediction", "enabled": True, "key_params": ['SGPT', 'ALCOHOL', 'Platelet_Count', 'BMI']},
    "ThyroidDisease": {"display_name": "Thyroid Disease", "importance_file": 'thyroid_disease_feature_importances_v12.joblib', "ml_score_col": "ThyroidDisease_RiskScore", "ml_zone_col": "ThyroidDisease_MLRiskZone", "rule_score_col": "ThyroidDisease_RiskScore_Calculated", "validation_col": "ThyroidDisease_Validation", "justification_col": "ThyroidDisease_Justification", "target_col": "ThyroidDisease_Target", "pred_col": "ThyroidDisease_ML_Prediction", "enabled": True, "key_params": ['TSH', 'Gender']},
    "MetabolicSyndrome": {"display_name": "Metabolic Syndrome", "importance_file": 'metabolic_syndrome_feature_importances_v12.joblib', "ml_score_col": "MetabolicSyndrome_RiskScore", "ml_zone_col": "MetabolicSyndrome_MLRiskZone", "rule_score_col": "MetabolicSyndrome_RiskScore_Calculated", "validation_col": "MetabolicSyndrome_Validation", "justification_col": "MetabolicSyndrome_Justification", "target_col": "MetabolicSyndrome_Target", "pred_col": "MetabolicSyndrome_ML_Prediction", "enabled": True, "key_params": ['BMI', 'TRIGLYCERIDE', 'HDL_CHOLESTEROL', 'BPSys', 'FBS']}
}
FINAL_DATA_FILE = 'Consolidated_Final_Predictions_v_thyroid.xlsx'
KNOWN_FEATURES_IN_SCORING_ENGINE = {
    'Age', 'BMI', 'BPSys', 'BPDia', 'CHOLESTEROL', 'LDL_CHOLESTEROL', 'HDL_CHOLESTEROL', 'TRIGLYCERIDE', 'FBS', 'CREATININE', 'VITAMIN_D3', 'ESR__1_HOUR', 'Hb_HEMOGLOBIN', 'RBC_Count', 'HbA1C', 'URIC_ACID', 'VITAMIN_B12', 'LDL/HDL_Ratio', 'CHOL/HDL_Ratio', 'SGPT', 'Platelet_Count', 'TSH', 'MCV', 'ALBUMIN_Serum', 'ECHO_LVEF_Value', 'Gender', 'SMOKER', 'ALCOHOL', 'CHEWER', 'ECG_RESULT', 'TMT_RESULT', 'ECHO_RESULT', 'SONOGRAPHY_RESULT', 'SUGAR', 'Pus_cells', 'Red_Cells', 'Casts', 'Epithelial_Cells', 'Crystals', 'BILESALT', 'BILEPIGMENT', 'Healthy_Lifestyle_Score', 'CardioMetabolic_Risk', 'MetabolicSyndrome_Indicator', 'ALBUMIN_Serum_Categorical', 'ALBUMIN_Serum_Numeric'
}

@st.cache_data
def load_and_prepare_data(file_path, config):
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[()]', '', regex=True)
        for col in ['LabId', 'Emp_No', 'B2BName', 'RegistrationDate']:
            if col in df.columns:
                if col == 'RegistrationDate':
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                else:
                    df[col] = df[col].astype(str).str.strip().replace('nan', '')
        
        summaries = {}
        for disease, info in config.items():
            required_cols = [info['ml_score_col'], info['validation_col'], info['justification_col']]
            if not all(col in df.columns for col in required_cols):
                st.warning(f"⚠️ Warning: Data for '{info['display_name']}' is incomplete. Related sections will be disabled.", icon="⚠️")
                config[disease]['enabled'] = False
                continue

            incorrect_df = df[df[info['validation_col']] == 'Incorrect']
            fp = incorrect_df[(incorrect_df[info['target_col']] == 0) & (incorrect_df[info['pred_col']] == 1)]
            fn = incorrect_df[(incorrect_df[info['target_col']] == 1) & (incorrect_df[info['pred_col']] == 0)]
            fp_reasons = analyze_justifications(fp[info['justification_col']])
            fn_reasons = analyze_justifications(fn[info['justification_col']])
            summaries[disease] = {"discrepancies": len(incorrect_df), "fp_count": len(fp), "fn_count": len(fn), "fp_reasons": fp_reasons, "fn_reasons": fn_reasons}
        return df, config, summaries
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: The main data file '{file_path}' was not found. Please ensure it is in the same directory as the app.", icon="🚨")
        return None, config, None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}", icon="🚨")
        return None, config, None

@st.cache_resource
def load_all_model_artifacts(config):
    artifacts = {}
    for disease, info in config.items():
        if not info.get('enabled', True): continue
        try:
            shap_values, one_hot_features, original_features = joblib.load(info["importance_file"])
            shap_values_class1 = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
            mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
            summary = pd.DataFrame({'feature': one_hot_features, 'mean_abs_shap': mean_abs_shap})
            artifacts[disease] = {'shap_summary': summary.sort_values('mean_abs_shap', ascending=False), 'original_features': sorted(original_features)}
        except FileNotFoundError:
            st.warning(f"⚠️ Warning: Model artifact '{info['importance_file']}' not found. Feature importance and related tools for '{info['display_name']}' will be disabled.", icon="⚠️")
            artifacts[disease] = {'shap_summary': None, 'original_features': []}
        except Exception as e:
            st.error(f"An error occurred loading artifact for {info['display_name']}: {e}", icon="🚨")
            artifacts[disease] = {'shap_summary': None, 'original_features': []}
    return artifacts

# --- START: REPLICATED & HARDENED LOGIC FROM JUPYTER NOTEBOOK ---
def calculate_heart_attack_risk_score(df, features=None):
    scores, reasons_list = [], []
    for i, row in df.iterrows():
        s, reasons = 0.0, []
        ldl = pd.to_numeric(row.get('LDL_CHOLESTEROL'), errors='coerce')
        chol = pd.to_numeric(row.get('CHOLESTEROL'), errors='coerce')
        bpsys = pd.to_numeric(row.get('BPSys'), errors='coerce')
        bmi = pd.to_numeric(row.get('BMI'), errors='coerce')
        trig = pd.to_numeric(row.get('TRIGLYCERIDE'), errors='coerce')
        age = pd.to_numeric(row.get('Age'), errors='coerce')
        vitd3 = pd.to_numeric(row.get('VITAMIN_D3'), errors='coerce')

        if row.get('NLP_DiseaseSelf_IHD_ACTIVE', 0) or row.get('NLP_PastHistorySelf_IHD_HISTORICAL', 0): s += 10.0; reasons.append("Self_History_IHD(10.0)")
        if row.get('NLP_Surgery_SURGERY_CABG_HISTORICAL', 0): s += 10.0; reasons.append("Past_CABG(10.0)")
        if row.get('NLP_Surgery_SURGERY_ANGIOPLASTY_HISTORICAL', 0): s += 9.0; reasons.append("Past_Angioplasty(9.0)")
        if pd.notna(ldl):
            if ldl >= 250: s += 6.0; reasons.append("LDL_Critical(>250)(6.0)")
            elif ldl >= 190: s += 4.0; reasons.append("LDL_High(>190)(4.0)")
            elif ldl >= 160: s += 2.5; reasons.append("LDL_Med(>160)(2.5)")
            elif ldl >= 130: s += 1.0; reasons.append("LDL_Low-Med(>130)(1.0)")
        if pd.notna(chol):
            if chol >= 350: s += 5.0; reasons.append("Chol_Critical(>350)(5.0)")
            elif chol >= 300: s += 3.0; reasons.append("Chol_High(>300)(3.0)")
            elif chol >= 240: s += 1.5; reasons.append("Chol_Med(>240)(1.5)")
        if pd.notna(bpsys):
            if bpsys >= 180: s += 4.0; reasons.append("BPSys_Critical(>=180)(4.0)")
            elif bpsys >= 160: s += 3.0; reasons.append("BPSys_High(>=160)(3.0)")
            elif bpsys >= 140: s += 1.5; reasons.append("BPSys_Med(>=140)(1.5)")
        if row.get('SMOKER') == 'Yes': s += 3.0; reasons.append("Smoker(3.0)")
        is_diabetic = row.get('NLP_DiseaseSelf_DIABETES_ACTIVE', 0) or row.get('NLP_DiseaseSelf_DIABETES', 0)
        if is_diabetic: s += 2.5; reasons.append("Active_DM(2.5)")
        if pd.notna(bmi):
            if bmi >= 35: s += 2.0; reasons.append("Obesity_II(>=35)(2.0)")
            elif bmi >= 30: s += 1.0; reasons.append("Obesity_I(>=30)(1.0)")
            elif bmi >= 25: s += 0.5; reasons.append("Overweight(>=25)(0.5)")
        if pd.notna(trig) and trig >= 200: s += 1.0; reasons.append("Trig_Med(>=200)(1.0)")
        if pd.notna(age) and age >= 60: s += 1.0; reasons.append("Age>=60(1.0)")
        family_ihd_flags = ['NLP_PastHistoryFather_IHD', 'NLP_PastHistoryMother_IHD']
        if any(row.get(flag, 0) for flag in family_ihd_flags): s += 1.0; reasons.append("Family_IHD(1.0)")
        if pd.notna(vitd3) and vitd3 < 20: s += 0.5; reasons.append("VitD3<20(0.5)")
        if row.get('ALCOHOL') == 'Yes': s += 0.5; reasons.append("Alcohol(0.5)")
        scores.append(max(0, s))
        reasons_list.append(" + ".join(reasons) if reasons else "No Significant Risk Factors")
    return pd.Series(scores, index=df.index), pd.Series(reasons_list, index=df.index)

def calculate_diabetes_risk_score(df, features=None):
    scores, reasons_list = [], []
    for i, row in df.iterrows():
        s, reasons = 0.0, []
        hba1c = pd.to_numeric(row.get('HbA1C'), errors='coerce')
        fbs = pd.to_numeric(row.get('FBS'), errors='coerce')
        bmi = pd.to_numeric(row.get('BMI'), errors='coerce')
        trig = pd.to_numeric(row.get('TRIGLYCERIDE'), errors='coerce')

        if pd.notna(hba1c):
            if hba1c >= 11.0: s += 12.0; reasons.append("HbA1c_Urgent(>=11)(12.0)")
            elif hba1c >= 9.0: s += 8.0; reasons.append("HbA1c_Critical(9-11)(8.0)")
            elif hba1c >= 6.5: s += 5.0; reasons.append("HbA1c_High(6.5-9)(5.0)")
            elif hba1c >= 5.7: s += 2.5; reasons.append("HbA1c_Med(5.7-6.4)(2.5)")
        if pd.notna(fbs):
            if fbs >= 200: s += 5.0; reasons.append("FBS_High(>=200)(5.0)")
            elif fbs >= 126: s += 3.0; reasons.append("FBS_Med(>=126)(3.0)")
            elif fbs >= 100: s += 1.5; reasons.append("FBS_PreDM(>=100)(1.5)")
        if row.get('NLP_DiseaseSelf_DIABETES_ACTIVE', 0): s += 6.0; reasons.append("Self_DM_Active(6.0)")
        if pd.notna(bmi):
            if bmi >= 35: s += 2.5; reasons.append("Obesity_II(>=35)(2.5)")
            elif bmi >= 30: s += 1.5; reasons.append("Obesity_I(>=30)(1.5)")
        family_dm_flags = ['NLP_PastHistoryFather_DIABETES', 'NLP_PastHistoryMother_DIABETES']
        if any(row.get(flag, 0) for flag in family_dm_flags): s += 1.5; reasons.append("Family_DM(1.5)")
        if pd.notna(trig) and trig >= 150: s += 1.0; reasons.append("Trig>=150(1.0)")
        scores.append(max(0, s))
        reasons_list.append(" + ".join(reasons) if reasons else "No Significant Risk Factors")
    return pd.Series(scores, index=df.index), pd.Series(reasons_list, index=df.index)

def calculate_kidney_disease_risk_score(df, features=None):
    scores, reasons_list = [], []
    for i, row in df.iterrows():
        s, reasons = 0.0, []
        creatinine = pd.to_numeric(row.get('CREATININE'), errors='coerce')
        hgb = pd.to_numeric(row.get('Hb_HEMOGLOBIN'), errors='coerce')

        if pd.notna(creatinine):
            if creatinine > 2.0: s += 10.0; reasons.append("Creat_Critical(>2.0)(10.0)")
            elif creatinine >= 1.8: s += 7.0; reasons.append("Creat_High(1.8-2.0)(7.0)")
            elif creatinine >= 1.3: s += 4.0; reasons.append("Creat_Med(1.3-1.8)(4.0)")
        if row.get('NLP_DiseaseSelf_CKD_ACTIVE', 0): s += 9.0; reasons.append("Self_CKD_Active(9.0)")
        if row.get('ALBUMIN_Serum_Categorical') in ['Present_Plus_2', 'Present_Plus_3']: s += 4.0; reasons.append(f"Proteinuria_High(4.0)")
        elif row.get('ALBUMIN_Serum_Categorical') in ['Trace', 'Present_Plus_1']: s += 2.0; reasons.append(f"Proteinuria_Med(2.0)")
        is_diabetic = row.get('NLP_DiseaseSelf_DIABETES_ACTIVE', 0) or row.get('NLP_DiseaseSelf_DIABETES', 0)
        if is_diabetic: s += 2.5; reasons.append("Active_DM(2.5)")
        if pd.notna(hgb):
            if hgb < 9.0: s += 3.0; reasons.append("Anemia_High(<9)(3.0)")
            elif hgb < 11.0: s += 1.5; reasons.append("Anemia_Med(<11)(1.5)")
        scores.append(max(0, s))
        reasons_list.append(" + ".join(reasons) if reasons else "No Significant Risk Factors")
    return pd.Series(scores, index=df.index), pd.Series(reasons_list, index=df.index)

def calculate_liver_disease_risk_score(df, features=None):
    scores, reasons_list = [], []
    for i, row in df.iterrows():
        s, reasons = 0.0, []
        sgpt = pd.to_numeric(row.get('SGPT'), errors='coerce')
        platelet = pd.to_numeric(row.get('Platelet_Count'), errors='coerce')
        bmi = pd.to_numeric(row.get('BMI'), errors='coerce')
        
        if pd.notna(sgpt):
            if sgpt > 150: s += 7.0; reasons.append("SGPT_Critical(>150)(7.0)")
            elif sgpt > 70: s += 5.0; reasons.append("SGPT_High(70-150)(5.0)")
            elif sgpt > 45: s += 2.5; reasons.append("SGPT_Med(45-70)(2.5)")
        if row.get('ALCOHOL') == 'Yes': s += 4.0; reasons.append("Alcohol_Yes(4.0)")
        if row.get('NLP_SONOGRAPHY_REMARK_SONO_Fatty_Liver_Grade_3', 0): s += 4.0; reasons.append("FattyLiver_G3(4.0)")
        elif row.get('NLP_SONOGRAPHY_REMARK_SONO_Fatty_Liver_Grade_2', 0): s += 2.5; reasons.append("FattyLiver_G2(2.5)")
        elif row.get('NLP_SONOGRAPHY_REMARK_SONO_Fatty_Liver_Generic', 0) or row.get('NLP_SONOGRAPHY_REMARK_SONO_Fatty_Liver_Grade_1', 0): 
            s += 1.0; reasons.append("FattyLiver_G1/Generic(1.0)")
        if pd.notna(platelet) and platelet < 150000: s += 2.0; reasons.append("Platelet<150k(2.0)")
        if pd.notna(bmi) and bmi >= 30: s += 1.0; reasons.append("Obesity_I(>=30)(1.0)")
        scores.append(max(0, s))
        reasons_list.append(" + ".join(reasons) if reasons else "No Significant Risk Factors")
    return pd.Series(scores, index=df.index), pd.Series(reasons_list, index=df.index)

def calculate_thyroid_disease_risk_score(df, features=None):
    scores, reasons_list = [], []
    for i, row in df.iterrows():
        s, reasons = 0.0, []
        tsh = pd.to_numeric(row.get('TSH'), errors='coerce')
        
        if row.get('NLP_DiseaseSelf_HYPOTHYROID_ACTIVE', 0) or row.get('NLP_DiseaseSelf_HYPERTHYROID_ACTIVE', 0): s += 8.0; reasons.append("Self_Thyroid_Active(8.0)")
        if pd.notna(tsh):
            if tsh > 20.0: s += 10.0; reasons.append("TSH_Urgent_High(>20)(10.0)")
            elif tsh > 10.0: s += 7.0; reasons.append("TSH_Critical_High(>10)(7.0)")
            elif tsh > 6.0: s += 4.0; reasons.append("TSH_High(>6)(4.0)")
            elif tsh > 4.5: s += 2.0; reasons.append("TSH_Med(>4.5)(2.0)")
            elif tsh < 0.1: s += 6.0; reasons.append("TSH_Critical_Low(<0.1)(6.0)")
            elif tsh < 0.4: s += 3.0; reasons.append("TSH_Med_Low(<0.4)(3.0)")
        family_thyroid_flags = ['NLP_PastHistoryMother_HYPOTHYROID', 'NLP_PastHistoryFather_HYPOTHYROID']
        if any(row.get(flag, 0) for flag in family_thyroid_flags): s += 2.0; reasons.append("Family_Hypo(2.0)")
        if pd.notna(row.get('Gender')) and row['Gender'] == 'Female': s += 1.0; reasons.append("Gender=Female(1.0)")
        scores.append(max(0, s))
        reasons_list.append(" + ".join(reasons) if reasons else "No Significant Risk Factors")
    return pd.Series(scores, index=df.index), pd.Series(reasons_list, index=df.index)

def calculate_metabolic_syndrome_risk_score(df, features=None):
    scores, reasons_list = [], []
    for i, row in df.iterrows():
        s, reasons = 0.0, []
        bmi = pd.to_numeric(row.get('BMI'), errors='coerce')
        trig = pd.to_numeric(row.get('TRIGLYCERIDE'), errors='coerce')
        hdl = pd.to_numeric(row.get('HDL_CHOLESTEROL'), errors='coerce')
        bpsys = pd.to_numeric(row.get('BPSys'), errors='coerce')
        fbs = pd.to_numeric(row.get('FBS'), errors='coerce')

        if pd.notna(bmi) and bmi >= 25: s += 2.0; reasons.append("Overweight(BMI>=25)(2.0)")
        if pd.notna(trig) and trig >= 150: s += 2.0; reasons.append("HighTrig(>=150)(2.0)")
        if pd.notna(hdl) and pd.notna(row.get('Gender')):
            if (row['Gender'] == 'Male' and hdl < 40) or \
               (row['Gender'] == 'Female' and hdl < 50):
                s += 2.0; reasons.append("LowHDL(2.0)")
        if pd.notna(bpsys) and bpsys >= 130: s += 2.0; reasons.append("HighBP(>=130)(2.0)")
        if pd.notna(fbs) and fbs >= 100: s += 2.0; reasons.append("HighFBS(>=100)(2.0)")
        if row.get('NLP_SONOGRAPHY_REMARK_SONO_Fatty_Liver_Grade_1', 0) or row.get('NLP_SONOGRAPHY_REMARK_SONO_Fatty_Liver_Generic', 0):
            s += 1.0; reasons.append("FattyLiver(1.0)")
        scores.append(max(0, s))
        reasons_list.append(" + ".join(reasons) if reasons else "No Significant Risk Factors")
    return pd.Series(scores, index=df.index), pd.Series(reasons_list, index=df.index)

SCORING_FUNCTION_MAP = {
    "HeartAttack": calculate_heart_attack_risk_score, "Diabetes": calculate_diabetes_risk_score,
    "KidneyDisease": calculate_kidney_disease_risk_score, "LiverDisease": calculate_liver_disease_risk_score,
    "ThyroidDisease": calculate_thyroid_disease_risk_score, "MetabolicSyndrome": calculate_metabolic_syndrome_risk_score
}
# --- END: REPLICATED & HARDENED LOGIC ---

def calculate_custom_index(row, param_weights):
    total_weighted_risk, contributions = 0, {}
    def normalize_risk(param, value):
        if pd.isna(value): return 0
        if isinstance(value, str):
            val_lower = value.lower()
            if val_lower in ['yes', 'abnormal'] or 'present' in val_lower: return 100
            return 0
        if isinstance(value, (int, float)):
            if param == 'BPSys': return min(100, max(0, (value - 100) / 0.8))
            if param == 'CHOLESTEROL': return min(100, max(0, (value - 120) / 1.8))
            if param == 'LDL_CHOLESTEROL': return min(100, max(0, (value - 70) / 1.2))
            if param == 'FBS': return min(100, max(0, (value - 70) / 1.3))
            if param == 'BMI': return min(100, max(0, (value - 18) / 0.22))
            return min(100, max(0, value))
        return 0
    for param, weight in param_weights.items():
        if param in row.index and pd.notna(row[param]) and weight > 0:
            risk_contribution = normalize_risk(param, row[param])
            if risk_contribution > 0:
                weighted_risk = risk_contribution * (weight / 100.0)
                total_weighted_risk += weighted_risk
                contributions[param] = weighted_risk
    return round(total_weighted_risk, 2), contributions

@st.cache_data
def assign_risk_zones_with_emojis(score_series):
    def classify(score):
        if pd.isna(score): return "0 - ⚪ N/A"
        if score >= 98: return '5 - 🌋 Urgent Risk'
        elif score >= 85: return '4 - 🚨 Critical Risk'
        elif score >= 60: return '3 - 🔴 High Risk'
        elif score >= 30: return '2 - 🟠 Medium Risk'
        else: return '1 - 🟢 Low Risk'
    return score_series.apply(classify)

def assign_custom_risk_zone_emoji(score):
    if score is None or pd.isna(score): return "0 - ⚪ N/A"
    if score <= 25: return '1 - 🟢 Low Risk'
    if score <= 50: return '2 - 🟠 Medium Risk'
    if score <= 75: return '3 - 🔴 High Risk'
    return '4 - 🚨 Critical Risk'

def clean_feature_name(feature):
    return feature.replace('NLP_', '').replace('_', ' ').replace(' nan', ' Unknown').replace(' Confirmed', '').strip().title()

def get_contribution_df(contributions_dict):
    if not contributions_dict: return None
    df = pd.DataFrame(list(contributions_dict.items()), columns=['Parameter', 'Contribution (Points)'])
    df['Parameter'] = df['Parameter'].apply(clean_feature_name)
    return df.sort_values('Contribution (Points)', ascending=False)

def generate_clinical_summary(record, config):
    summary_parts = []
    age = record.get('Age', 'N/A')
    gender = record.get('Gender', 'Unknown').lower()
    summary_parts.append(f"This is a **{age}-year-old {gender}** patient.")
    vitals_parts = []
    if pd.notna(record.get('BMI')): vitals_parts.append(f"a BMI of **{record.get('BMI'):.1f}**")
    if pd.notna(record.get('BPSys')) and pd.notna(record.get('BPDia')): vitals_parts.append(f"blood pressure of **{record.get('BPSys')}/{record.get('BPDia')} mmHg**")
    if vitals_parts: summary_parts.append(f"Key vitals include { ' and '.join(vitals_parts) }.")
    high_risk_diseases = [info['display_name'] for key, info in config.items() if info.get('enabled', False) and not str(record.get(info['ml_zone_col'] + "_Sortable", '1 -')).startswith(('1 -', '2 -', '0 -'))]
    if high_risk_diseases: summary_parts.append(f"The **ML models** identified elevated risk for: **{', '.join(high_risk_diseases)}**.")
    else: summary_parts.append("The **ML models** did not identify elevated risk for any of the monitored conditions.")
    disagreements = [info['display_name'] for key, info in config.items() if info.get('enabled', False) and record.get(info['validation_col']) == 'Incorrect']
    if disagreements: summary_parts.append(f"**Note:** A disagreement between the rule-based score and the ML model was noted for **{', '.join(disagreements)}**.")
    if 'ML_Risk_Override_Reason' in record.index:
        override_reason = record.get('ML_Risk_Override_Reason', '')
        if pd.notna(override_reason) and override_reason:
            summary_parts.append(f"**Safety Net Applied:** The ML risk assessment was automatically adjusted by the system's safety protocols. Details are in the specific disease sections below.")
    if pd.notna(record.get('Justification_Note')) and record.get('Justification_Note'): summary_parts.append(f"**Clinical Note:** {record.get('Justification_Note')}")
    return " ".join(summary_parts)

def analyze_justifications(justification_series):
    if not isinstance(justification_series, pd.Series) or justification_series.empty: return {}
    causes = []
    pattern = re.compile(r"Key drivers for risk: (.*?)\.")
    for just_text in justification_series.dropna():
        match = pattern.search(just_text)
        if match:
            factors_text = match.group(1)
            if factors_text:
                factors = [factor.strip() for factor in factors_text.split(',')]
                causes.extend(factors)
    return pd.Series(causes).value_counts().head(5).to_dict()
#</editor-fold>

# --- 4. Dashboard Display Functions ---
def display_executive_dashboard(df_processed, enabled_diseases):
    st.title("🏢 Executive Command Center")
    st.markdown("A high-level, strategic overview of workforce health.")
    st.subheader("Filters")
    
    if 'exec_b2b' not in st.session_state: st.session_state.exec_b2b = "All"
    if 'exec_lab_id' not in st.session_state: st.session_state.exec_lab_id = ""
    if 'exec_emp_no' not in st.session_state: st.session_state.exec_emp_no = ""

    col1, col2, col3, col4 = st.columns([3, 3, 3, 1])
    with col1:
        b2b_options = ["All"]
        if 'B2BName' in df_processed.columns and df_processed['B2BName'].nunique() > 0: b2b_options.extend(sorted(df_processed[df_processed['B2BName'] != '']['B2BName'].unique().tolist()))
        st.selectbox("Filter by B2B Client", b2b_options, key='exec_b2b')
    with col2: st.text_input("Search by Lab ID", key='exec_lab_id')
    with col3: st.text_input("Search by Employee No", key='exec_emp_no')
    with col4:
        st.write("") 
        st.write("")
        apply_filters = st.button("Search / Apply", use_container_width=True)

    df_filtered = df_processed.copy()
    if st.session_state.exec_b2b != "All": df_filtered = df_filtered[df_filtered['B2BName'] == st.session_state.exec_b2b]
    if st.session_state.exec_lab_id: df_filtered = df_filtered[df_filtered['LabId'].str.contains(st.session_state.exec_lab_id, case=False, na=False)]
    if st.session_state.exec_emp_no: df_filtered = df_filtered[df_filtered['Emp_No'].str.contains(st.session_state.exec_emp_no, case=False, na=False)]
    st.divider()

    if not df_filtered.empty:
        st.subheader("Key Performance Indicators (KPIs)")
        total_employees = len(df_filtered)
        risk_cols = [info['ml_zone_col'] + "_Sortable" for info in enabled_diseases.values()]
        high_critical_mask = df_filtered[risk_cols].apply(lambda x: x.str.startswith(('3 -', '4 -', '5 -'))).any(axis=1)
        at_risk_count = high_critical_mask.sum()
        at_risk_percent = (at_risk_count / total_employees) * 100 if total_employees > 0 else 0

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Employees Analyzed", f"{total_employees:,}", help="Total number of employees matching the current filter criteria.")
        kpi2.metric("Workforce at High/Critical/Urgent Risk", f"{at_risk_percent:.1f}%", help="Percentage of filtered employees with a 'High', 'Critical', or 'Urgent' risk rating in any disease category.")
        risk_counts = {}
        for info in enabled_diseases.values():
            col = info['ml_zone_col'] + "_Sortable"
            risk_counts[info['display_name']] = df_filtered[col].str.startswith(('3 -', '4 -', '5 -')).sum()
        top_risk = max(risk_counts, key=risk_counts.get) if risk_counts else "N/A"
        kpi3.metric("Top Health Risk", top_risk, help="The disease category with the highest number of employees in the 'High', 'Critical', or 'Urgent' risk tiers.")
    else: st.warning("No data matches the current filter criteria.")
    st.subheader("Patient Data Explorer")
    display_global_patient_table(df_filtered, enabled_diseases)

def display_global_patient_table(df_display, enabled_diseases):
    show_all_cols = st.checkbox("Show All Data Columns", key="global_show_all")
    default_cols_map = {'LabId': 'LabId', 'Emp_No': 'Emp_No', 'B2BName': 'B2BName', 'Overall_Health_MLRiskZone_Sortable': 'Overall Health'}
    for dk, di in enabled_diseases.items(): default_cols_map[di['ml_zone_col'] + '_Sortable'] = di['display_name']
    display_cols = df_display.columns.tolist() if show_all_cols else [k for k in default_cols_map.keys() if k in df_display.columns]
    df_view = df_display[display_cols].rename(columns=default_cols_map)
    st.dataframe(df_view)

def display_clinician_dashboard(df_processed, enabled_diseases, model_artifacts, discrepancy_summaries, df_full):
    st.title("🩺 Clinician Deep-Dive")
    st.sidebar.title("Clinician Controls")
    
    if 'current_patient' not in st.session_state: st.session_state.current_patient = None
    
    if st.session_state.current_patient != st.session_state.get('clinician_search_term'):
        st.session_state.current_patient = st.session_state.get('clinician_search_term')
        keys_to_clear = ['custom_scores', 'custom_contributions', 'verification_results', 'is_calculated', 'param_weights', 'overall_weights']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
    
    def clear_clinician_search():
        st.session_state.clinician_search_term = ""
        st.session_state.current_patient = ""
        keys_to_clear = ['custom_scores', 'custom_contributions', 'verification_results', 'is_calculated', 'param_weights', 'overall_weights']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.toast("Patient selection and custom weights cleared.", icon="✅")

    def reset_all_weights():
        keys_to_clear = ['param_weights', 'overall_weights', 'custom_scores', 'custom_contributions', 'is_calculated']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.toast("Custom weights have been reset.", icon="🔄")

    st.sidebar.header("Patient Search")
    search_col = st.sidebar.selectbox("Search By", ['LabId', 'Emp_No', 'B2BName'], key="clinician_search_col")
    unique_values = [""] + sorted(df_processed[search_col].astype(str).unique())
    search_term = st.sidebar.selectbox(f"Select a value for {search_col}", unique_values, key="clinician_search_term")
    st.sidebar.button("Clear Patient Selection", on_click=clear_clinician_search, use_container_width=True)
    single_patient_selected = search_term != ""

    patient_record_df = df_processed[df_processed[search_col].astype(str) == search_term] if single_patient_selected else pd.DataFrame()

    if not single_patient_selected or patient_record_df.empty:
        if single_patient_selected and patient_record_df.empty:
            st.warning(f"No patient found with {search_col} = '{search_term}'. Please try another search.", icon="⚠️")
        
        st.sidebar.subheader("Global Patient Filters")
        risk_levels_to_filter = ['5 - 🌋 Urgent Risk', '4 - 🚨 Critical Risk', '3 - 🔴 High Risk', '2 - 🟠 Medium Risk', '1 - 🟢 Low Risk']
        selected_risks = st.sidebar.multiselect("Filter by Risk Level", options=risk_levels_to_filter, format_func=lambda x: x.split(' - ')[-1])
        df_display = df_processed.copy()
        if selected_risks:
            risk_cols = [info['ml_zone_col'] + "_Sortable" for info in enabled_diseases.values()] + ['Overall_Health_MLRiskZone_Sortable']
            mask = df_display[risk_cols].isin(selected_risks).any(axis=1)
            df_display = df_display[mask]
        st.info("Select a patient from the sidebar to begin a deep-dive analysis.")
        display_global_patient_table(df_display, enabled_diseases)
        
        with st.expander("🧠 Global Model Insights & Summaries", expanded=True):
            tabs = st.tabs(["Prediction Distribution", "Model Discrepancy Report", "Top Feature Importance"])
            with tabs[0]:
                st.markdown("#### Prediction Distribution Summary (All Patients)")
                zone_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk', 'Urgent Risk', 'Modeling Skipped', 'N/A']
                all_distributions = {**enabled_diseases, "Overall_Health": {"display_name": "Overall Health", "ml_zone_col": "Overall_Health_MLRiskZone"}}
                for key, info in all_distributions.items():
                    st.markdown(f"**{info['display_name']}**")
                    zone_col = info['ml_zone_col']
                    if zone_col in df_full.columns:
                        counts = df_full[zone_col].value_counts().reindex(zone_order).fillna(0).astype(int).reset_index(name='Count').rename(columns={'index': 'Risk Zone'})
                        st.dataframe(counts, use_container_width=True)
            with tabs[1]:
                st.markdown("#### Model Discrepancy Report")
                for key, info in enabled_diseases.items():
                    summary = discrepancy_summaries.get(key, {})
                    st.markdown(f"##### --- {info['display_name'].upper()} ---")
                    st.markdown(f"- Total Disagreements: **{summary.get('discrepancies', 'N/A')}**")
                    if summary.get('fp_count', 0) > 0: st.markdown(f"  - False Positives (ML High, Rule Low): **{summary.get('fp_count', 'N/A')}** (Top Drivers: `{summary.get('fp_reasons', {})}`)")
                    if summary.get('fn_count', 0) > 0: st.markdown(f"  - False Negatives (ML Low, Rule High): **{summary.get('fn_count', 'N/A')}** (Top Drivers: `{summary.get('fn_reasons', {})}`)")
            with tabs[2]:
                st.markdown("#### Top 15 Most Important Features Per Model (from .joblib)")
                for disease_key, d_info in enabled_diseases.items():
                    st.write(f"##### Key Factors for {d_info['display_name']}")
                    artifacts = model_artifacts.get(disease_key, {})
                    if artifacts and artifacts.get('shap_summary') is not None:
                        top_15 = artifacts['shap_summary'].head(15).copy()
                        top_15['feature'] = top_15['feature'].apply(clean_feature_name)
                        top_15.rename(columns={'feature': 'Feature', 'mean_abs_shap': 'Mean SHAP Impact'}, inplace=True)
                        st.dataframe(top_15.set_index('Feature').style.format({'Mean SHAP Impact': "{:.4f}"}), use_container_width=True)
                    else: st.warning(f"Could not load feature importance data for {d_info['display_name']}.")
    else:
        record = patient_record_df.iloc[0]
        
        #<editor-fold desc="Clinician Sidebar Controls">
        with st.sidebar.expander("🔬 What-If Analysis (Custom Index)", expanded=False):
            st.info("Adjust weights to create a custom health index for the selected patient.")
            st.caption("Note: Weights for all features are remembered, even when switching between Simple/Advanced views.")
            st.subheader("Overall Health Index Weights")
            if 'overall_weights' not in st.session_state: st.session_state.overall_weights = {d: (100 // len(enabled_diseases)) if enabled_diseases else 0 for d in enabled_diseases}
            temp_overall_weights = {dk: st.number_input(f"{di['display_name']}", 0, 100, st.session_state.overall_weights.get(dk, 0), key=f"overall_{dk}") for dk, di in enabled_diseases.items()}
            total_overall_weight = sum(temp_overall_weights.values())
            st.metric("Total Weight", f"{total_overall_weight}%", delta=f"{100-total_overall_weight}% Required", delta_color="inverse" if total_overall_weight == 100 else "normal")
            if total_overall_weight == 100: st.session_state.overall_weights = temp_overall_weights

            st.subheader("Disease-Specific Parameter Weights")
            view_mode = st.sidebar.radio("Parameter Weighting View", ["Simple (Top Features)", "Advanced (All Features)"], key="view_mode", horizontal=True)
            if 'param_weights' not in st.session_state: st.session_state.param_weights = {}
            if 'custom_scores' not in st.session_state: st.session_state.custom_scores = {}
            if 'custom_contributions' not in st.session_state: st.session_state.custom_contributions = {}
            if 'is_calculated' not in st.session_state: st.session_state.is_calculated = {}

            for disease_key, d_info in enabled_diseases.items():
                with st.expander(f"Customize {d_info['display_name']} Score"):
                    artifacts = model_artifacts.get(disease_key, {})
                    all_features = artifacts.get('original_features', [])
                    if not all_features:
                        st.info("Weight customization disabled (model artifacts not found).")
                        continue
                    if disease_key not in st.session_state.param_weights: st.session_state.param_weights[disease_key] = {f: 0 for f in all_features}
                    
                    features_to_show = all_features
                    if view_mode == "Simple (Top Features)" and artifacts.get('shap_summary') is not None:
                        shap_summary_df, simple_view_features = artifacts['shap_summary'], []
                        top_features_onehot = shap_summary_df['feature'].head(15).tolist()
                        for one_hot_feat in top_features_onehot:
                            parent_feat = next((orig_feat for orig_feat in all_features if one_hot_feat.startswith(orig_feat)), one_hot_feat)
                            if parent_feat not in simple_view_features: simple_view_features.append(parent_feat)
                        features_to_show = sorted(simple_view_features)
                    
                    for feat in features_to_show:
                        current_weight = st.session_state.param_weights[disease_key].get(feat, 0)
                        new_weight = st.number_input(clean_feature_name(feat), 0, 100, current_weight, key=f"param_{disease_key}_{feat}")
                        if new_weight != current_weight:
                            st.session_state.param_weights[disease_key][feat] = new_weight
                            st.session_state.is_calculated[disease_key] = False
                    
                    total_param_weight = sum(st.session_state.param_weights[disease_key].values())
                    st.metric(f"{d_info['display_name']} Total", f"{total_param_weight}%", delta=f"{100-total_param_weight}% Required", delta_color="inverse" if total_param_weight == 100 else "normal")
                    
                    if st.button(f"Calculate {d_info['display_name']} Index", key=f"calc_{disease_key}", use_container_width=True, disabled=(total_param_weight != 100)):
                        score, contributions = calculate_custom_index(record, st.session_state.param_weights[disease_key])
                        st.session_state.custom_scores[disease_key] = score
                        st.session_state.custom_contributions[disease_key] = contributions
                        st.session_state.is_calculated[disease_key] = True
            
            if st.button("🔄 Reset Custom Weights", use_container_width=True, on_click=reset_all_weights): st.rerun()

        with st.sidebar.expander("🔍 Rule-Based Score Verification", expanded=True):
            st.info("Select a disease to run a live calculation of the system's rule-based score for this patient, providing a transparent audit of the logic.")
            disease_options = {info['display_name']: key for key, info in enabled_diseases.items()}
            selected_disease_display = st.selectbox("Select Disease to Verify", options=[""] + list(disease_options.keys()), key="verify_disease_select")
            if st.button("Verify Score", use_container_width=True, disabled=(not selected_disease_display)):
                disease_key = disease_options[selected_disease_display]
                scoring_function = SCORING_FUNCTION_MAP.get(disease_key)
                if scoring_function:
                    scores, reasons = scoring_function(record.to_frame().T)
                    missing_params = [p for p in DISEASE_MODELS_INFO[disease_key]['key_params'] if pd.isna(record.get(p))]
                    st.session_state.verification_results = {'disease': selected_disease_display, 'score': scores.iloc[0], 'breakdown': reasons.iloc[0], 'missing': missing_params}
                else: st.session_state.verification_results = {'disease': selected_disease_display, 'score': 'Error', 'breakdown': 'Scoring function not found.', 'missing': []}
            if 'verification_results' in st.session_state:
                results = st.session_state.verification_results
                st.subheader(f"Verification for: {results['disease']}")
                st.metric("Live Calculated Score", f"{results['score']:.2f}")
                st.markdown("**Contributing Factors (from Notebook Logic):**")
                st.text_area("Breakdown", results['breakdown'], height=150, disabled=True, key="verify_breakdown")
                if results['missing']:
                    st.warning(f"Note: Key data for this calculation was missing: {', '.join(results['missing'])}")
        
        with st.sidebar.expander("🎛️ Verify Scoring Engine Features"):
            st.write("Verifies features from the ML model have a rule in the custom index engine.")
            for disease_key, d_info in enabled_diseases.items():
                st.write(f"**{d_info['display_name']}**")
                model_features = set(model_artifacts.get(disease_key, {}).get('original_features', []))
                if model_features:
                    missing = [f for f in model_features if not (f in KNOWN_FEATURES_IN_SCORING_ENGINE or f.startswith("NLP_"))]
                    if not missing: st.success("✅ All features implemented.")
                    else: st.warning(f"⚠️ Missing Logic: `{', '.join(missing)}`")
                else: st.info("Verification skipped; model artifacts not loaded.")
        #</editor-fold>

        st.header(f"🔬 Detailed Analysis for Lab ID: {record.get('LabId', 'N/A')}")
        st.markdown(f"**Patient Summary:** {generate_clinical_summary(record, enabled_diseases)}")
        with st.expander("View Full Patient Data"): st.dataframe(record.to_frame('Value'), use_container_width=True)
        st.divider()

        st.subheader("📊 Overall Health Score Comparison")
        overall_custom_score = 0
        calculated_diseases = []
        if 'overall_weights' in st.session_state and sum(st.session_state.overall_weights.values()) == 100 and st.session_state.get('custom_scores'):
            weighted_score_sum = 0
            for dk, score in st.session_state.custom_scores.items():
                weight = st.session_state.overall_weights.get(dk, 0)
                if weight > 0:
                    weighted_score_sum += score * (weight / 100.0)
                    calculated_diseases.append(enabled_diseases[dk]['display_name'])
            overall_custom_score = weighted_score_sum
        overall_help_text = f"(Based on: {', '.join(calculated_diseases)})" if calculated_diseases else "(Calculate at least one custom disease index)"
        col1, col2 = st.columns(2)
        col1.metric("Overall AI-Powered ML Score", f"{record.get('Overall_Health_RiskScore', 0):.1f}", help="Average of the 6 AI model scores, aligned to the final risk zone.")
        col2.metric("Overall Custom Health Index", f"{overall_custom_score:.1f}", help=overall_help_text)

        st.subheader("🩺 Disease-Specific Breakdown")
        for disease_key, info in enabled_diseases.items():
            ml_score_val = record.get(info['ml_score_col'], 0)
            ml_zone_val_text = record.get(info['ml_zone_col'] + "_Sortable", "0 - ⚪ N/A")
            custom_score = st.session_state.custom_scores.get(disease_key)
            custom_zone = assign_custom_risk_zone_emoji(custom_score)
            is_calculated = st.session_state.is_calculated.get(disease_key, False)
            
            with st.expander(f"**{info['display_name']}** — ML Zone: {ml_zone_val_text} | Custom Index Zone: {custom_zone}", expanded=True):
                if 'ML_Risk_Override_Reason' in record.index:
                    override_reason = record.get('ML_Risk_Override_Reason', '')
                    if pd.notna(override_reason) and disease_key in override_reason:
                        if 'Downgraded_FP_Catch' in override_reason: st.info("ℹ️ **Safety Net Override:** This patient's ML risk was automatically downgraded due to consistently normal clinical markers.")
                        elif 'Upgraded_FN_Catch' in override_reason: st.warning("⚠️ **Safety Net Override:** This patient's ML risk was automatically upgraded to High Risk based on a critical rule-based score.")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Rule-Based vs. ML Model**")
                    st.metric(label="Rule-Based Score (Notebook)", value=f"{record.get(info['rule_score_col'], 0):.1f}")
                    breakdown_col = f"{disease_key}_Calculation_Breakdown"
                    if breakdown_col in record and pd.notna(record[breakdown_col]): st.caption(f"Reason: {record[breakdown_col]}")
                    st.metric(label="ML Model Score (Final & Aligned)", value=f"{ml_score_val:.1f}", delta=ml_zone_val_text.split(' - ')[-1])
                    val_status = record.get(info['validation_col'], 'N/A')
                    if val_status == 'Correct': st.success("✅ Agreement")
                    elif val_status == 'Incorrect':
                        st.warning("⚠️ Disagreement Found")
                        justification = record.get(info['justification_col'], 'No justification available.')
                        if pd.notna(justification) and justification: st.info(f"_{justification}_")
                with col2:
                    st.markdown("**Custom Health Index (What-If)**")
                    score_display = f"{custom_score:.1f}" if (custom_score is not None and is_calculated) else "Not Calculated"
                    if custom_score is not None and not is_calculated:
                        st.warning("Weights have changed. Click 'Calculate' in the sidebar to update.")
                    elif custom_score is None:
                        st.info("Use the 'What-If Analysis' controls in the sidebar to calculate a custom score.")
                    
                    st.metric(label="Health Index Score / Zone", value=score_display, delta=custom_zone.split(' - ')[-1], delta_color="off", help="Scale: 0-100+. Zones: Low (0-25), Medium (26-50), High (51-75), Critical (>75)")
                    if custom_score is not None and is_calculated:
                        st.markdown("**Top Factors Driving Custom Score**")
                        contributions = st.session_state.custom_contributions.get(disease_key, {})
                        contribution_df = get_contribution_df(contributions)
                        if contribution_df is not None and not contribution_df.empty: st.dataframe(contribution_df, use_container_width=True)
                        else: st.info("No factors significantly contributed.")

def display_employee_dashboard(employee_record):
    st.title(f"My Health Report")
    st.success("🔒 This report is private and visible only to you.", icon="🔒")
    st.markdown("A personalized overview of your recent health check-up.")
    st.info("This tool provides educational summaries and data visualization. For medical advice, please consult the company clinician.", icon="ℹ️")
    st.divider()
    st.header("Your Key Numbers at a Glance")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Body Mass Index (BMI)", f"{employee_record.get('BMI', 0):.1f}")
    with col2: st.metric("Blood Pressure", f"{employee_record.get('BPSys', 'N/A')} / {employee_record.get('BPDia', 'N/A')}")
    with col3: st.metric("Fasting Blood Sugar (FBS)", f"{employee_record.get('FBS', 0):.1f}")
    st.header("Your Overall Health Status")
    overall_status_text = employee_record.get("Overall_Health_MLRiskZone_Sortable", "N/A").split(' - ')[-1]
    st.info(f"The automated health models have placed your overall health status in the **{overall_status_text}** category based on your results.")

def display_login_screen(df):
    st.title("Login to Your Dashboard")
    st.markdown("Please select your role to continue.")
    role = st.selectbox("Select Your Role", ["", "Executive / Manager", "Clinician", "Employee"])
    if role == "Employee":
        with st.form("employee_login"):
            emp_no_input = st.text_input("Enter Your Employee Number")
            lab_id_input = st.text_input("Enter Your Lab ID")
            b2b_name_input = st.text_input("Enter Your Company / B2B Name")
            submitted = st.form_submit_button("View My Report")
            if submitted:
                emp_no = emp_no_input.strip()
                lab_id = lab_id_input.strip()
                b2b_name = b2b_name_input.strip()
                if emp_no and lab_id and b2b_name:
                    df_comp = df.copy()
                    df_comp['B2BName_comp'] = df_comp['B2BName'].str.strip().str.lower()
                    b2b_name_comp = b2b_name.lower()
                    
                    emp_record = df_comp[(df_comp['Emp_No'] == emp_no) & (df_comp['LabId'] == lab_id) & (df_comp['B2BName_comp'] == b2b_name_comp)]
                    if not emp_record.empty:
                        original_index = emp_record.index[0]
                        st.session_state.logged_in_role = "Employee"
                        st.session_state.user_data = df.loc[original_index]
                        st.rerun()
                    else: st.error("Invalid credentials. Please check your Employee No, Lab ID, and Company Name.")
                else: st.warning("Please enter your Employee Number, Lab ID, and Company Name.")
    elif role in ["Executive / Manager", "Clinician"]:
        if st.button(f"Login as {role}"):
            st.session_state.logged_in_role = role
            st.rerun()

# --- 5. Main Application Execution ---
def password_protected_app():
    # --- This is the main body of your application ---
    df_full, ACTIVE_DISEASE_CONFIG, discrepancy_summaries = load_and_prepare_data(FINAL_DATA_FILE, DISEASE_MODELS_INFO.copy())
    model_artifacts = load_all_model_artifacts(ACTIVE_DISEASE_CONFIG)

    if df_full is not None:
        enabled_diseases = {dk: di for dk, di in ACTIVE_DISEASE_CONFIG.items() if di.get('enabled', True)}
        df_processed = df_full.copy()
        for info in enabled_diseases.values():
            if info['ml_score_col'] in df_processed.columns:
                df_processed[info['ml_zone_col'] + "_Sortable"] = assign_risk_zones_with_emojis(df_processed[info['ml_score_col']])
        if 'Overall_Health_RiskScore' in df_processed.columns:
            df_processed['Overall_Health_MLRiskZone_Sortable'] = assign_risk_zones_with_emojis(df_processed['Overall_Health_RiskScore'])

        if 'logged_in_role' not in st.session_state:
            display_login_screen(df_processed)
        else:
            logout_popover = st.sidebar.popover("Logout", use_container_width=True)
            if logout_popover.button("Confirm Logout", use_container_width=True, type="primary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("You have been successfully logged out.")
                st.rerun()
            
            if st.session_state.logged_in_role == "Employee":
                display_employee_dashboard(st.session_state.user_data)
            elif st.session_state.logged_in_role == "Clinician":
                display_clinician_dashboard(df_processed, enabled_diseases, model_artifacts, discrepancy_summaries, df_full)
            elif st.session_state.logged_in_role == "Executive / Manager":
                display_executive_dashboard(df_processed, enabled_diseases)
    else:
        st.error("Application cannot start because the main data file failed to load.")


# --- This is the new password checking logic ---
if __name__ == "__main__":
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    # Check if secrets are loaded, which indicates it's running on the cloud
    if hasattr(st, 'secrets') and "APP_PASSWORD" in st.secrets:
        # Cloud execution path
        if not st.session_state.password_correct:
            st.title("Organizational Health Intelligence Platform")
            st.header("Secure Login")
            password_input = st.text_input("Enter Password to Access", type="password")
            
            if password_input == st.secrets["APP_PASSWORD"]:
                st.session_state.password_correct = True
                st.rerun()
            elif password_input:
                st.error("The password you entered is incorrect.")
        else:
            password_protected_app()
    else:
        # Local execution path (no password needed)
        password_protected_app()