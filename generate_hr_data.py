"""
===============================================================
  Nigerian Financial Institution — Synthetic HR Dataset
  Generator v2
===============================================================
  Generates a realistic 5,000-row HR dataset for a Nigerian
  bank covering the period January 2015 – December 2024.

  Key design principles:
  - Latent risk score drives feature values and attrition
    probability to produce realistic statistical correlations
  - Attrition rate calibrated to ~25% (Nigerian banking sector)
  - Salary ranges in Naira aligned to Nigerian banking grades
  - 37 departments reflecting real Nigerian bank structure
  - Nigerian states, geopolitical zones, and education fields

  Target correlations with Attrition:
  - EngagementScore:           ~-0.35
  - JobSatisfaction:           ~-0.27
  - WorkLifeBalance:           ~-0.24
  - ManagerRelationship:       ~-0.19
  - OvertimeFrequency:         ~+0.15
  - YearsSinceLastPromotion:   ~+0.13

  Usage:
      python generate_hr_data.py

  Output:
      nigerian_bank_hr_dataset_final.csv
===============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

N = 5000

DEPARTMENTS = [
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
    "Information Technology (IT) Department",
    "Core Banking Applications Support",
    "Information & Cyber Security Department",
    "Data Analytics & Business Intelligence",
    "Customer Service / Customer Experience Department",
    "Human Resources & Talent Management",
    "Finance & Corporate Strategy Department",
    "Corporate Communications & Marketing",
    "Procurement & Supply Chain Department",
    "Facilities & Administrative Services",
    "Corporate Social Responsibility (CSR) Unit"
]

DEPT_WEIGHTS = np.array([
    7,5,5,4,2,2,2,1,3,2,2,1,2,7,2,2,
    3,2,2,3,3,2,2,2,2,2,3,2,2,2,6,3,
    2,1,1,2,1
], dtype=float)
DEPT_WEIGHTS /= DEPT_WEIGHTS.sum()

IT_DEPARTMENTS = {
    "Information Technology (IT) Department",
    "Core Banking Applications Support",
    "Information & Cyber Security Department",
    "Data Analytics & Business Intelligence",
    "Digital Banking Department",
    "Alternative Channels Department"
}

HIGH_OT_DEPARTMENTS = {
    "Treasury & Markets Department",
    "Internal Audit Department",
    "Regulatory Compliance Department",
    "Information & Cyber Security Department"
}

NIGERIAN_STATES = [
    "Lagos","Abuja (FCT)","Rivers","Ogun","Oyo","Kano","Kaduna",
    "Delta","Anambra","Enugu","Imo","Edo","Akwa Ibom","Cross River",
    "Abia","Ebonyi","Kogi","Kwara","Niger","Benue","Plateau",
    "Nasarawa","Adamawa","Bauchi","Gombe","Taraba","Yobe","Borno",
    "Jigawa","Kebbi","Sokoto","Zamfara","Katsina","Osun","Ondo",
    "Ekiti","Bayelsa"
]
STATE_WEIGHTS = np.array([
    25,12,7,6,5,3,3,4,3,3,3,3,2,2,
    2,1,1,1,1,1,1,1,1,1,1,.5,.5,.5,
    .5,.5,.5,.5,.5,1,1,1,.5
], dtype=float)
STATE_WEIGHTS /= STATE_WEIGHTS.sum()

EDUCATION_FIELDS = [
    "Accounting","Finance","Economics","Banking & Finance",
    "Business Administration","Computer Science","Information Technology",
    "Engineering","Statistics","Mathematics","Law","Others"
]
EDU_FIELD_WEIGHTS = np.array(
    [15,15,12,10,10,8,7,7,5,4,4,3], dtype=float
)
EDU_FIELD_WEIGHTS /= EDU_FIELD_WEIGHTS.sum()


def random_date(start, end):
    if start >= end:
        return start
    return start + timedelta(days=random.randint(0,(end-start).days))


def satisfaction_score(risk, p1, p2, p3, p4):
    noise = np.random.normal(risk, 0.5)
    shift = np.clip(noise * 0.28, -0.22, 0.22)
    p = np.array([p1+shift, p2+shift*0.5, p3-shift*0.7, p4-shift])
    p = np.clip(p, 0.02, 0.90); p /= p.sum()
    return int(np.random.choice([1,2,3,4], p=p))


def attrition_probability(i, **kw):
    S = 0.38
    p = 0.02
    p += {1:0.30,2:0.18,3:0.04,4:-0.04}[kw['js']] * S
    p += {1:0.22,2:0.12,3:0.02,4:-0.05}[kw['wlb']] * S
    p += {1:0.18,2:0.10,3:0.00,4:-0.06}[kw['mr']] * S
    p += {1:0.12,2:0.06,3:0.00,4:-0.04}[kw['es']] * S
    eng = kw['eng']
    p += (0.22 if eng<25 else 0.14 if eng<40 else 0.06 if eng<55
          else -0.10 if eng>75 else 0) * S
    p += {1:0.18,2:0.06,3:0.00,4:-0.08,5:-0.12}[kw['lv']] * S
    ysp = kw['ysp']
    p += (0.15 if ysp>4 else 0.10 if ysp>3 else 0.05 if ysp>2 else 0) * S
    p += {'Always':0.18,'Often':0.10,'Sometimes':0.00,
          'Rarely':-0.02,'Never':-0.04}[kw['ot']] * S
    lsi = kw['lsi']
    p += (0.10 if lsi<2 else 0.05 if lsi<4 else -0.06 if lsi>10 else 0) * S
    p += (-0.04 if kw['bonus']=='Yes' else 0.07) * S
    tt = kw['tt']
    p += (0.07 if tt==0 else 0.03 if tt==1 else -0.05 if tt>=4 else 0) * S
    age = kw['age']
    p += (0.12 if age<26 else 0.07 if age<30 else -0.10 if age>50 else 0) * S
    yac = kw['yac']
    p += (0.12 if yac<1 else 0.06 if yac<2 else -0.10 if yac>10 else 0) * S
    dist = kw['dist']
    p += (0.10 if dist>300 else 0.05 if dist>150 else 0) * S
    if kw['js']<=2 and kw['wlb']<=2: p += 0.10*S
    if kw['lv']==1 and kw['ot'] in ['Often','Always']: p += 0.08*S
    if kw['ysp']>3 and kw['js']<=2: p += 0.10*S
    if kw['bonus']=='No' and kw['lsi']<3: p += 0.07*S
    return float(np.clip(p, 0.03, 0.92))


print("Generating synthetic HR dataset...")

risk = np.random.normal(0, 1, N)
dept_arr   = np.random.choice(DEPARTMENTS, size=N, p=DEPT_WEIGHTS)
job_levels = np.random.choice([1,2,3,4,5], size=N,
                               p=[0.12,0.33,0.30,0.18,0.07])
emp_types  = np.where(job_levels==1, "Contract", "Permanent")

ages = np.array([
    np.random.randint(22,35) if l==1 else np.random.randint(24,38) if l==2
    else np.random.randint(28,45) if l==3 else np.random.randint(35,52) if l==4
    else np.random.randint(42,58) for l in job_levels
])
genders = np.random.choice(["Male","Female"], size=N, p=[0.58,0.42])
marital = [np.random.choice(
    ["Single","Married","Divorced"],
    p=[0.75,0.23,0.02] if a<27 else [0.35,0.60,0.05] if a<35 else [0.15,0.75,0.10]
) for a in ages]
states     = np.random.choice(NIGERIAN_STATES, size=N, p=STATE_WEIGHTS)
edus       = [
    np.random.choice([1,2,3],p=[0.25,0.35,0.40]) if l==1
    else np.random.choice([2,3,4],p=[0.15,0.65,0.20]) if l==2
    else np.random.choice([3,4,5],p=[0.50,0.43,0.07]) if l==3
    else np.random.choice([3,4,5],p=[0.30,0.55,0.15]) if l==4
    else np.random.choice([3,4,5],p=[0.20,0.55,0.25]) for l in job_levels
]
edu_fields = np.random.choice(EDUCATION_FIELDS, size=N, p=EDU_FIELD_WEIGHTS)

report_date = datetime(2024,12,31)
hire_dates  = [random_date(datetime(2015,1,1),datetime(2024,6,30)) for _ in range(N)]
yac = [max(0.1,round((report_date-hd).days/365.25,1)) for hd in hire_dates]
yir = [max(0.1,min(round(np.random.uniform(0.3,y*0.85+0.3),1),y)) for y in yac]
twy = [max(y,round(y+np.random.uniform(0,max(0,ages[i]-23-y)),1)) for i,y in enumerate(yac)]

js  = np.array([satisfaction_score(risk[i],0.22,0.30,0.27,0.21) for i in range(N)])
wlb = np.array([satisfaction_score(risk[i],0.23,0.32,0.27,0.18) for i in range(N)])
mr  = np.array([satisfaction_score(risk[i],0.24,0.33,0.26,0.17) for i in range(N)])
es  = np.array([satisfaction_score(risk[i],0.23,0.31,0.27,0.19) for i in range(N)])
eng = np.array([int(min(100,max(1,round(
    ((js[i]+wlb[i]+mr[i]+es[i])/16)*100+np.random.uniform(-4,4)-max(0,risk[i])*6
)))) for i in range(N)])

ysp = [min(yac[i],max(0.0,round(
    np.random.uniform(0,min(yac[i],6))+max(0,risk[i])*1.0,1
))) for i in range(N)]
npromo = [
    0 if l==1 else np.random.choice([0,1],p=[0.5,0.5]) if l==2
    else np.random.choice([1,2,3],p=[0.4,0.4,0.2]) if l==3
    else np.random.choice([2,3,4],p=[0.3,0.4,0.3]) if l==4
    else np.random.choice([3,4,5,6],p=[0.2,0.3,0.3,0.2]) for l in job_levels
]
incomes = np.array([
    np.random.randint(250000,451000)
    if job_levels[i]==1 and dept_arr[i] in IT_DEPARTMENTS and np.random.random()<0.15
    else np.random.randint(150000,251000) if job_levels[i]==1
    else np.random.randint(500000,801000) if job_levels[i]==2
    else np.random.randint(800000,1501000) if job_levels[i]==3
    else np.random.randint(1500000,3501000) if job_levels[i]==4
    else np.random.randint(3500000,8001000) for i in range(N)
])
sbands = [{1:"Contract",2:"Junior",3:"Mid",4:"Senior",5:"Executive"}[l] for l in job_levels]
lsi = [
    round(max(0,np.random.uniform(3,15)-max(0,risk[i])*1.5),1)
    if emp_types[i]=='Permanent' and job_levels[i]<5
    else round(np.random.uniform(5,20),1) if job_levels[i]==5
    else round(np.random.uniform(0,5),1) for i in range(N)
]
bonus = [
    "No" if emp_types[i]=="Contract" and np.random.random()<0.90
    else "Yes" if job_levels[i]>=4 and np.random.random()<0.80
    else np.random.choice(
        ["Yes","No"],
        p=[max(0.10,0.55-max(0,risk[i])*0.10),
           min(0.90,0.45+max(0,risk[i])*0.10)]
    ) for i in range(N)
]
pr = []
for i,l in enumerate(job_levels):
    adj = max(0,risk[i])*0.06
    base = ([0.10+adj,0.25+adj*0.5,0.45-adj*0.8,0.20-adj] if l<=2
            else [0.05+adj,0.20+adj*0.5,0.50-adj*0.8,0.25-adj] if l==3
            else [0.02+adj,0.13+adj*0.5,0.50-adj*0.8,0.35-adj])
    p = np.clip(base,0.02,0.90); p/=p.sum()
    pr.append(int(np.random.choice([1,2,3,4],p=p)))

tt  = [int(np.random.choice([0,1,2],p=[0.40,0.40,0.20]))
       if emp_types[i]=="Contract"
       else int(np.random.choice([0,1,2,3,4,5],p=[0.05,0.20,0.35,0.25,0.10,0.05]))
       for i in range(N)]
tgt = [round(np.random.uniform(20,50),1) if p==1
       else round(np.random.uniform(50,70),1) if p==2
       else round(np.random.uniform(70,90),1) if p==3
       else round(np.random.uniform(88,100),1) for p in pr]

dist = [
    np.random.randint(1,60) if s=="Lagos"
    else np.random.randint(30,120) if s in ["Ogun","Oyo"]
    else np.random.randint(400,600) if s=="Abuja (FCT)"
    else np.random.randint(150,800) for s in states
]

ot_arr = []
for i,dept in enumerate(dept_arr):
    shift = np.clip(risk[i]*0.22,-0.18,0.22)
    base  = ([0.05,0.15,0.30,0.35,0.15] if dept in HIGH_OT_DEPARTMENTS
             else [0.05,0.20,0.35,0.30,0.10] if job_levels[i]>=4
             else [0.15,0.30,0.35,0.15,0.05])
    p = np.array([base[0]-shift,base[1]-shift*0.5,
                  base[2],base[3]+shift*0.5,base[4]+shift])
    p = np.clip(p,0.02,0.90); p/=p.sum()
    ot_arr.append(np.random.choice(
        ["Never","Rarely","Sometimes","Often","Always"],p=p))

bt = [
    np.random.choice(["None","Occasional","Frequent"],p=[0.20,0.45,0.35]) if l>=4
    else np.random.choice(["None","Occasional","Frequent"],p=[0.35,0.50,0.15]) if l==3
    else np.random.choice(["None","Occasional","Frequent"],p=[0.65,0.30,0.05])
    for l in job_levels
]
rw = [
    "Yes" if dept_arr[i] in IT_DEPARTMENTS and np.random.random()<0.60
    else "Yes" if hire_dates[i].year>=2022 and np.random.random()<0.35
    else "No" for i in range(N)
]

att = [
    "Yes" if np.random.random() < attrition_probability(
        i, js=js[i], wlb=wlb[i], mr=mr[i], es=es[i], eng=eng[i],
        lv=job_levels[i], ysp=ysp[i], ot=ot_arr[i], lsi=lsi[i],
        bonus=bonus[i], tt=tt[i], age=ages[i], yac=yac[i], dist=dist[i]
    ) else "No" for i in range(N)
]

REASONS = [
    "BETTER OPPORTUNITY","CAREER PROGRESSION","PERSONAL REASONS",
    "RELOCATION","ACADEMICS","CAREER BREAK","RETIREMENT",
    "TERMINATION","HEALTH REASONS","ENTREPRENEURSHIP"
]
rfx = [
    np.random.choice(REASONS,p=[0.10,0.10,0.15,0.05,0.02,0.05,0.40,0.05,0.05,0.03])
    if att[i]=="Yes" and ages[i]>55
    else np.random.choice(REASONS,p=[0.35,0.25,0.10,0.08,0.08,0.05,0.01,0.05,0.02,0.01])
    if att[i]=="Yes" and job_levels[i]==1
    else np.random.choice(REASONS,p=[0.30,0.28,0.12,0.08,0.05,0.05,0.02,0.05,0.03,0.02])
    if att[i]=="Yes" else None for i in range(N)
]

exits, statuses = [], []
for i in range(N):
    if att[i]=="No":
        exits.append(None); statuses.append("Active")
    else:
        me = hire_dates[i]+timedelta(days=60)
        mx = datetime(2024,12,31)
        if me>=mx: mx=me+timedelta(days=30)
        ed = random_date(me,mx)
        exits.append(ed.strftime("%Y-%m-%d"))
        r = rfx[i]
        statuses.append(
            "Retired" if r=="RETIREMENT"
            else "Terminated" if r=="TERMINATION"
            else "Resigned")

df = pd.DataFrame({
    "EmployeeID":              [f"EMP-{str(i+1).zfill(5)}" for i in range(N)],
    "HireDate":                [d.strftime("%Y-%m-%d") for d in hire_dates],
    "ExitDate":                exits,
    "EmploymentStatus":        statuses,
    "EmploymentType":          emp_types,
    "Age":                     ages,
    "Gender":                  genders,
    "MaritalStatus":           marital,
    "StateOfOrigin":           states,
    "Education":               edus,
    "EducationField":          edu_fields,
    "Department":              dept_arr,
    "JobLevel":                job_levels,
    "YearsAtCompany":          yac,
    "YearsInRole":             yir,
    "TotalWorkingYears":       twy,
    "YearsSinceLastPromotion": ysp,
    "NumberOfPromotions":      npromo,
    "MonthlyIncome":           incomes,
    "SalaryBand":              sbands,
    "LastSalaryIncreasePct":   lsi,
    "BonusReceived":           bonus,
    "PerformanceRating":       pr,
    "TrainingTimesLastYear":   tt,
    "TargetsMetPct":           tgt,
    "JobSatisfaction":         js,
    "WorkLifeBalance":         wlb,
    "ManagerRelationship":     mr,
    "EnvironmentSatisfaction": es,
    "EngagementScore":         eng,
    "DistanceFromHome":        dist,
    "OvertimeFrequency":       ot_arr,
    "BusinessTravel":          bt,
    "RemoteWorkOption":        rw,
    "Attrition":               att,
    "Reason_for_Leaving":      rfx
})

df = df.sort_values("HireDate").reset_index(drop=True)
df["EmployeeID"] = [f"EMP-{str(i+1).zfill(5)}" for i in range(N)]
df.to_csv("nigerian_bank_hr_dataset_final.csv", index=False)

df["Ab"]  = (df["Attrition"]=="Yes").astype(int)
df["OTe"] = df["OvertimeFrequency"].map(
    {"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Always":4})
cols = ["JobSatisfaction","WorkLifeBalance","ManagerRelationship",
        "EnvironmentSatisfaction","EngagementScore","JobLevel",
        "MonthlyIncome","YearsSinceLastPromotion",
        "DistanceFromHome","OTe","Age","LastSalaryIncreasePct"]
corrs = df[cols+["Ab"]].corr()["Ab"].drop("Ab").sort_values().round(3)

print("=" * 58)
print("  DATASET GENERATION COMPLETE")
print("=" * 58)
print(f"  Output          : nigerian_bank_hr_dataset_final.csv")
print(f"  Total records   : {len(df):,}")
print(f"  Attrition rate  : {df['Ab'].mean():.1%}")
print(f"  Active          : {(df['EmploymentStatus']=='Active').sum():,}")
print(f"  Resigned        : {(df['EmploymentStatus']=='Resigned').sum():,}")
print()
print("  SALARY RANGES BY JOB LEVEL:")
print(df.groupby("JobLevel")["MonthlyIncome"].agg(Min="min",Max="max").to_string())
print()
print("  FEATURE CORRELATIONS WITH ATTRITION:")
print(corrs.to_string())
print("=" * 58)
