"""
Configuration for readmission triage model.
Single source of truth for feature definitions and settings.

Section 1 (ALL_*): Raw dataset schema — never changes unless the CSV changes.
                 EDA functions use these to see everything.

Section (EDA_DROP): Features to exclude based on EDA findings.
                    Add columns here with a comment explaining why.

Section 3 (NUMERIC_COLS, CATEGORICAL_COLS, FEATURE_COLS):
                    What the model actually trains on.
                    Derived automatically from Layer 1 - Layer 2.
"""

RANDOM_SEED = 42

RAW_DATA_PATH = "data/raw/diabetic_data.csv"

LABEL_COL = "readmitted"

# Columns to drop before EDA, we don't need these at all for the model.
DROP_COLS = [
    "encounter_id",
    "patient_nbr",
]

# Section 1: Initial feature set, we perform EDA on this and then decide what to keep/drop.

ALL_NUMERIC = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# NOTE: This also includes ID columns like admission_type_id etc.
ALL_CATEGORICAL = [
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]

ALL_FEATURES = ALL_NUMERIC + ALL_CATEGORICAL

# Section 2: This config is used after EDA.

EDA_DROP = [
    "examide",                   
    "citoglipton",               
    "acetohexamide",            
    "troglitazone",              
    "tolbutamide",               
    "tolazamide",                
    "miglitol",                  
    "chlorpropamide",            
    "glipizide-metformin",       
    "glimepiride-pioglitazone",  
    "metformin-rosiglitazone",   
    "metformin-pioglitazone",    
    "weight",                    
]

# Engineered features (added in feature_engineering.py)

ENGINEERED_NUMERIC = [
    "utilization_index",
    "med_up_count",
    "med_down_count",
    "med_steady_count"
]

# Section 3: Model feature set (auto-derived from Layer 1 and Layer 2)

NUMERIC_COLS = [c for c in ALL_NUMERIC if c not in EDA_DROP] + ENGINEERED_NUMERIC
CATEGORICAL_COLS = [c for c in ALL_CATEGORICAL if c not in EDA_DROP]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def print_experiment_config():
    print(__doc__)
    print(f"RANDOM_SEED -> {RANDOM_SEED}")
    print(f"RAW_DATA_PATH -> {RAW_DATA_PATH}")
    print(f"LABEL_COL -> {LABEL_COL}")
    print(f"DROP_COLS -> {DROP_COLS}\n")
    print(f"ALL_NUMERIC ({len(ALL_NUMERIC)}) -> {ALL_NUMERIC}\n")
    print(f"ALL_CATEGORICAL ({len(ALL_CATEGORICAL)}) -> {ALL_CATEGORICAL}\n")
    print(f"EDA_DROP ({len(EDA_DROP)}) -> {EDA_DROP}\n")
    print(f"ENGINEERED_NUMERIC -> {ENGINEERED_NUMERIC}\n")
    print(f"NUMERIC_COLS ({len(NUMERIC_COLS)}) -> {NUMERIC_COLS}\n")
    print(f"CATEGORICAL_COLS ({len(CATEGORICAL_COLS)}) -> {CATEGORICAL_COLS}\n")
    print(f"FEATURE_COLS ({len(FEATURE_COLS)}) -> {FEATURE_COLS}\n")


if __name__ == "__main__":
    print_experiment_config()
