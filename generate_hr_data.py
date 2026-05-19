import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

N = 5000

#  REFERENCE DATA 

departments = [
    "Retail Banking Department", "Commercial Banking Department",
    "Corporate Banking Department", "SME Banking Department",
    "Private Banking & Wealth Management", "Public Sector Department",
    "Institutional Banking Department", "Agricultural Finance Department",
    "Treasury & Markets Department", "Foreign Exchange (FX) Operations",
    "Trade Services Department", "International Remittances Department",
    "Asset Management Department", "Branch Operations Department",
    "Cash Management & Vault Operations", "Clearing & Settlement Department",
    "Digital Banking Department", "Alternative Channels Department",
    "Card Services Department", "Central Operations Department",
    "Risk Management Department", "Credit Administration Department",
    "Legal Department", "Regulatory Compliance Department",
    "Internal Audit Department", "Internal Control & Anti-Fraud Unit",
    "Information Technology (IT) Department", "Core Banking Applications Support",
    "Information & Cyber Security Department", "Data Analytics & Business Intelligence",
    "Customer Service / Customer Experience Department",
    "Human Resources & Talent Management", "Finance & Corporate Strategy Department",
    "Corporate Communications & Marketing", "Procurement & Supply Chain Department",
    "Facilities & Administrative Services", "Corporate Social Responsibility (CSR) Unit"
]

dept_weights = np.array([
    7, 5, 5, 4, 2, 2, 2, 1, 3, 2, 2, 1, 2, 7, 2, 2,
    3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 6, 3,
    2, 1, 1, 2, 1
], dtype=float)
dept_weights /= dept_weights.sum()

it_departments = {
    "Information Technology (IT) Department",
    "Core Banking Applications Support",
    "Information & Cyber Security Department",
    "Data Analytics & Business Intelligence",
    "Digital Banking Department",
    "Alternative Channels Department"
}

nigerian_states = [
    "Lagos", "Abuja (FCT)", "Rivers", "Ogun", "Oyo", "Kano", "Kaduna",
    "Delta", "Anambra", "Enugu", "Imo", "Edo", "Akwa Ibom", "Cross River",
    "Abia", "Ebonyi", "Kogi", "Kwara", "Niger", "Benue", "Plateau",
    "Nasarawa", "Adamawa", "Bauchi", "Gombe", "Taraba", "Yobe", "Borno",
    "Jigawa", "Kebbi", "Sokoto", "Zamfara", "Katsina", "Osun", "Ondo",
    "Ekiti", "Bayelsa"
]

state_weights = np.array([
    25, 12, 7, 6, 5, 3, 3, 4, 3, 3, 3, 3, 2, 2,
    2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 0.5
], dtype=float)
state_weights /= state_weights.sum()

education_fields = [
    "Accounting", "Finance", "Economics", "Banking & Finance",
    "Business Administration", "Computer Science", "Information Technology",
    "Engineering", "Statistics", "Mathematics", "Law", "Others"
]
edu_field_weights = np.array([15, 15, 12, 10, 10, 8, 7, 7, 5, 4, 4, 3], dtype=float)
edu_field_weights /= edu_field_weights.sum()

#  CORE COLUMNS 

employee_ids = [f"EMP-{str(i).zfill(5)}" for i in range(1, N+1)]

dept_arr = np.random.choice(departments, size=N, p=dept_weights)

job_levels = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.12, 0.33, 0.30, 0.18, 0.07])

employment_types = np.where(job_levels == 1, "Contract", "Permanent")

# Salary
monthly_incomes = np.zeros(N, dtype=int)
for i in range(N):
    lv, dept = job_levels[i], dept_arr[i]
    if lv == 1:
        if dept in it_departments and np.random.random() < 0.15:
            monthly_incomes[i] = np.random.randint(250000, 451000)
        else:
            monthly_incomes[i] = np.random.randint(150000, 251000)
    elif lv == 2: monthly_incomes[i] = np.random.randint(500000, 801000)
    elif lv == 3: monthly_incomes[i] = np.random.randint(800000, 1501000)
    elif lv == 4: monthly_incomes[i] = np.random.randint(1500000, 3501000)
    else:         monthly_incomes[i] = np.random.randint(3500000, 8001000)

salary_bands = [
    {1: "Contract", 2: "Junior", 3: "Mid", 4: "Senior", 5: "Executive"}[l]
    for l in job_levels
]

# Age
ages = np.zeros(N, dtype=int)
for i, lv in enumerate(job_levels):
    if lv == 1:   ages[i] = np.random.randint(22, 35)
    elif lv == 2: ages[i] = np.random.randint(24, 38)
    elif lv == 3: ages[i] = np.random.randint(28, 45)
    elif lv == 4: ages[i] = np.random.randint(35, 52)
    else:         ages[i] = np.random.randint(42, 58)

genders = np.random.choice(["Male", "Female"], size=N, p=[0.58, 0.42])

marital_statuses = []
for age in ages:
    if age < 27:   ms = np.random.choice(["Single","Married","Divorced"], p=[0.75,0.23,0.02])
    elif age < 35: ms = np.random.choice(["Single","Married","Divorced"], p=[0.35,0.60,0.05])
    else:          ms = np.random.choice(["Single","Married","Divorced"], p=[0.15,0.75,0.10])
    marital_statuses.append(ms)

state_origins = np.random.choice(nigerian_states, size=N, p=state_weights)

educations = []
for lv in job_levels:
    if lv == 1:   edu = np.random.choice([1,2,3], p=[0.25,0.35,0.40])
    elif lv == 2: edu = np.random.choice([2,3,4], p=[0.15,0.65,0.20])
    elif lv == 3: edu = np.random.choice([3,4,5], p=[0.50,0.43,0.07])
    elif lv == 4: edu = np.random.choice([3,4,5], p=[0.30,0.55,0.15])
    else:         edu = np.random.choice([3,4,5], p=[0.20,0.55,0.25])
    educations.append(edu)

edu_fields_arr = np.random.choice(education_fields, size=N, p=edu_field_weights)

#  DATES 

report_date = datetime(2024, 12, 31)

def rand_date(start, end):
    if start >= end:
        return start
    return start + timedelta(days=random.randint(0, (end - start).days))

hire_dates = [rand_date(datetime(2015, 1, 1), datetime(2024, 6, 30)) for _ in range(N)]

#  TENURE & CAREER 

years_at_company = [max(0.1, round((report_date - hd).days / 365.25, 1)) for hd in hire_dates]

years_in_roles = [
    max(0.1, min(round(np.random.uniform(0.3, yac * 0.85 + 0.3), 1), yac))
    for yac in years_at_company
]

total_working_years = [
    max(yac, round(yac + np.random.uniform(0, max(0, ages[i] - 23 - yac)), 1))
    for i, yac in enumerate(years_at_company)
]

years_since_promo = [
    round(np.random.uniform(0, min(yac, 6)), 1)
    for yac in years_at_company
]

num_promotions = []
for lv in job_levels:
    if lv == 1:   p = 0
    elif lv == 2: p = np.random.choice([0,1], p=[0.5,0.5])
    elif lv == 3: p = np.random.choice([1,2,3], p=[0.4,0.4,0.2])
    elif lv == 4: p = np.random.choice([2,3,4], p=[0.3,0.4,0.3])
    else:         p = np.random.choice([3,4,5,6], p=[0.2,0.3,0.3,0.2])
    num_promotions.append(p)

#  COMPENSATION 

last_salary_increase = []
for i, lv in enumerate(job_levels):
    if employment_types[i] == "Contract": pct = round(np.random.uniform(0, 5), 1)
    elif lv == 5:                          pct = round(np.random.uniform(5, 20), 1)
    else:                                  pct = round(np.random.uniform(3, 15), 1)
    last_salary_increase.append(pct)

bonus_received = []
for i, lv in enumerate(job_levels):
    if employment_types[i] == "Contract": b = np.random.choice(["Yes","No"], p=[0.10,0.90])
    elif lv >= 4:                          b = np.random.choice(["Yes","No"], p=[0.80,0.20])
    else:                                  b = np.random.choice(["Yes","No"], p=[0.50,0.50])
    bonus_received.append(b)

#  PERFORMANCE ─

performance_ratings = []
for lv in job_levels:
    if lv <= 2: pr = np.random.choice([1,2,3,4], p=[0.10,0.25,0.45,0.20])
    elif lv==3: pr = np.random.choice([1,2,3,4], p=[0.05,0.20,0.50,0.25])
    else:       pr = np.random.choice([1,2,3,4], p=[0.02,0.13,0.50,0.35])
    performance_ratings.append(pr)

training_times = []
for i, lv in enumerate(job_levels):
    if employment_types[i] == "Contract": tt = np.random.choice([0,1,2], p=[0.40,0.40,0.20])
    elif lv == 5:                          tt = np.random.choice([1,2,3,4], p=[0.20,0.35,0.30,0.15])
    else:                                  tt = np.random.choice([0,1,2,3,4,5], p=[0.05,0.20,0.35,0.25,0.10,0.05])
    training_times.append(tt)

targets_met = []
for pr in performance_ratings:
    if pr == 1:   pct = round(np.random.uniform(20, 50), 1)
    elif pr == 2: pct = round(np.random.uniform(50, 70), 1)
    elif pr == 3: pct = round(np.random.uniform(70, 90), 1)
    else:         pct = round(np.random.uniform(88, 100), 1)
    targets_met.append(pct)

#  SATISFACTION 

job_satisfactions    = np.random.choice([1,2,3,4], size=N, p=[0.12,0.22,0.40,0.26])
work_life_balances   = np.random.choice([1,2,3,4], size=N, p=[0.10,0.25,0.42,0.23])
manager_relationships= np.random.choice([1,2,3,4], size=N, p=[0.08,0.20,0.45,0.27])
env_satisfactions    = np.random.choice([1,2,3,4], size=N, p=[0.10,0.22,0.42,0.26])

engagement_scores = [
    int(min(100, max(1, round(
        ((job_satisfactions[i] + work_life_balances[i] +
          manager_relationships[i] + env_satisfactions[i]) / 16) * 100
        + np.random.uniform(-5, 5)
    )))
    ) for i in range(N)
]

#  WORK PATTERNS 

distance_from_home = []
for state in state_origins:
    if state == "Lagos":           dist = np.random.randint(1, 60)
    elif state in ["Ogun","Oyo"]:  dist = np.random.randint(30, 120)
    elif state == "Abuja (FCT)":   dist = np.random.randint(400, 600)
    else:                          dist = np.random.randint(150, 800)
    distance_from_home.append(dist)

high_ot = {"Treasury & Markets Department","Internal Audit Department",
           "Regulatory Compliance Department","Information & Cyber Security Department"}

overtime_freqs = []
for i, dept in enumerate(dept_arr):
    lv = job_levels[i]
    if dept in high_ot:
        of = np.random.choice(["Never","Rarely","Sometimes","Often","Always"], p=[0.05,0.15,0.30,0.35,0.15])
    elif lv >= 4:
        of = np.random.choice(["Never","Rarely","Sometimes","Often","Always"], p=[0.05,0.20,0.35,0.30,0.10])
    else:
        of = np.random.choice(["Never","Rarely","Sometimes","Often","Always"], p=[0.15,0.30,0.35,0.15,0.05])
    overtime_freqs.append(of)

business_travels = []
for lv in job_levels:
    if lv >= 4:   bt = np.random.choice(["None","Occasional","Frequent"], p=[0.20,0.45,0.35])
    elif lv == 3: bt = np.random.choice(["None","Occasional","Frequent"], p=[0.35,0.50,0.15])
    else:         bt = np.random.choice(["None","Occasional","Frequent"], p=[0.65,0.30,0.05])
    business_travels.append(bt)

remote_work = []
for i, dept in enumerate(dept_arr):
    if dept in it_departments:
        rw = np.random.choice(["Yes","No"], p=[0.60,0.40])
    elif hire_dates[i].year >= 2022:
        rw = np.random.choice(["Yes","No"], p=[0.35,0.65])
    else:
        rw = np.random.choice(["Yes","No"], p=[0.15,0.85])
    remote_work.append(rw)

#  ATTRITION 

attrition_flags = []
for i in range(N):
    prob = 0.15
    if job_satisfactions[i] <= 2:     prob += 0.12
    if work_life_balances[i] == 1:    prob += 0.08
    if manager_relationships[i] == 1: prob += 0.06
    if job_levels[i] == 1:            prob += 0.08
    if job_levels[i] >= 4:            prob -= 0.06
    if last_salary_increase[i] < 3:   prob += 0.04
    if years_since_promo[i] > 3:      prob += 0.04
    if distance_from_home[i] > 100:   prob += 0.03
    prob = min(max(prob, 0.05), 0.70)
    attrition_flags.append("Yes" if np.random.random() < prob else "No")

leaving_reasons_pool = [
    "BETTER OPPORTUNITY","CAREER PROGRESSION","PERSONAL REASONS",
    "RELOCATION","ACADEMICS","CAREER BREAK","RETIREMENT",
    "TERMINATION","HEALTH REASONS","ENTREPRENEURSHIP"
]

reasons_for_leaving = []
for i, att in enumerate(attrition_flags):
    if att == "Yes":
        age, lv = ages[i], job_levels[i]
        if age > 55:
            r = np.random.choice(leaving_reasons_pool, p=[0.10,0.10,0.15,0.05,0.02,0.05,0.40,0.05,0.05,0.03])
        elif lv == 1:
            r = np.random.choice(leaving_reasons_pool, p=[0.35,0.25,0.10,0.08,0.08,0.05,0.01,0.05,0.02,0.01])
        else:
            r = np.random.choice(leaving_reasons_pool, p=[0.30,0.28,0.12,0.08,0.05,0.05,0.02,0.05,0.03,0.02])
        reasons_for_leaving.append(r)
    else:
        reasons_for_leaving.append(None)

exit_dates_final, emp_statuses_final = [], []
for i, att in enumerate(attrition_flags):
    if att == "Yes":
        hd = hire_dates[i]
        min_exit = hd + timedelta(days=60)
        max_exit = datetime(2024, 12, 31)
        if min_exit >= max_exit:
            max_exit = min_exit + timedelta(days=30)
        exit_d = rand_date(min_exit, max_exit)
        exit_dates_final.append(exit_d.strftime("%Y-%m-%d"))
        reason = reasons_for_leaving[i]
        if reason == "RETIREMENT":   emp_statuses_final.append("Retired")
        elif reason == "TERMINATION": emp_statuses_final.append("Terminated")
        else:                         emp_statuses_final.append("Resigned")
    else:
        exit_dates_final.append(None)
        emp_statuses_final.append("Active")

#  ASSEMBLE DATAFRAME 

df = pd.DataFrame({
    "EmployeeID":              employee_ids,
    "HireDate":                [d.strftime("%Y-%m-%d") for d in hire_dates],
    "ExitDate":                exit_dates_final,
    "EmploymentStatus":        emp_statuses_final,
    "EmploymentType":          employment_types,
    "Age":                     ages,
    "Gender":                  genders,
    "MaritalStatus":           marital_statuses,
    "StateOfOrigin":           state_origins,
    "Education":               educations,
    "EducationField":          edu_fields_arr,
    "Department":              dept_arr,
    "JobLevel":                job_levels,
    "YearsAtCompany":          years_at_company,
    "YearsInRole":             years_in_roles,
    "TotalWorkingYears":       total_working_years,
    "YearsSinceLastPromotion": years_since_promo,
    "NumberOfPromotions":      num_promotions,
    "MonthlyIncome":           monthly_incomes,
    "SalaryBand":              salary_bands,
    "LastSalaryIncreasePct":   last_salary_increase,
    "BonusReceived":           bonus_received,
    "PerformanceRating":       performance_ratings,
    "TrainingTimesLastYear":   training_times,
    "TargetsMetPct":           targets_met,
    "JobSatisfaction":         job_satisfactions,
    "WorkLifeBalance":         work_life_balances,
    "ManagerRelationship":     manager_relationships,
    "EnvironmentSatisfaction": env_satisfactions,
    "EngagementScore":         engagement_scores,
    "DistanceFromHome":        distance_from_home,
    "OvertimeFrequency":       overtime_freqs,
    "BusinessTravel":          business_travels,
    "RemoteWorkOption":        remote_work,
    "Attrition":               attrition_flags,
    "Reason_for_Leaving":      reasons_for_leaving
})

# Sort by HireDate so EmployeeIDs are chronologically consistent
df = df.sort_values("HireDate").reset_index(drop=True)
df["EmployeeID"] = [f"EMP-{str(i).zfill(5)}" for i in range(1, N+1)]

df.to_csv("nigerian_bank_hr_dataset.csv", index=False)

print("=" * 55)
print("  NIGERIAN BANK HR DATASET — GENERATION COMPLETE")
print("=" * 55)
print(f"  Total records        : {len(df):,}")
print(f"  Total columns        : {len(df.columns)}")
print(f"  Attrition rate       : {(df['Attrition']=='Yes').mean():.1%}")
print(f"  Active employees     : {(df['EmploymentStatus']=='Active').sum():,}")
print(f"  Resigned             : {(df['EmploymentStatus']=='Resigned').sum():,}")
print(f"  Terminated           : {(df['EmploymentStatus']=='Terminated').sum():,}")
print(f"  Retired              : {(df['EmploymentStatus']=='Retired').sum():,}")
print(f"  Date range           : 2015 – 2024")
print("=" * 55)
print("\n  SALARY VALIDATION BY JOB LEVEL:")
print(df.groupby(["JobLevel","SalaryBand"])["MonthlyIncome"]
      .agg(Min="min", Max="max", Mean="mean")
      .round(0).to_string())
print("\n  CONTRACT vs PERMANENT SPLIT:")
print(df["EmploymentType"].value_counts().to_string())
print("\n  TOP 5 DEPARTMENTS BY HEADCOUNT:")
print(df["Department"].value_counts().head(5).to_string())
print("\n  ATTRITION BY JOB LEVEL:")
print(df.groupby("JobLevel")["Attrition"]
      .apply(lambda x: f"{(x=='Yes').mean():.1%}").to_string())
print("\n  TOP LEAVING REASONS:")
print(df["Reason_for_Leaving"].value_counts().head(5).to_string())
