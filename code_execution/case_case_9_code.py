from z3 import *

# -----------------------------------------------------------------------------
# 建立 Z3 Optimize 物件（支援軟性約束）
opt = Optimize()

# -----------------------------------------------------------------------------
# 變數定義與上下界設定（單位：金額＝百萬元，次數＝次，人數＝人）
# -----------------------------------------------------------------------------
# ——— 內部控制核心指標 ———
internal_control_established   = Bool('internal_control_established')   # 已建立制度（§45-1）
internal_control_effective     = Bool('internal_control_effective')     # 能有效運作（§45-1）
audit_staff_qualified          = Bool('audit_staff_qualified')          # 稽核人員資格符合法規（實施辦法§12）

# ——— 稽核人力配置 ———
branch_count       = Int('branch_count')        # 分支機構數
auditor_count      = Int('auditor_count')       # 專任稽核人員數
opt.add(branch_count >= 0, branch_count <= 500)
opt.add(auditor_count >= 0, auditor_count <= 500)

# ——— 航空器租賃 ———
aircraft_lease_cost        = Real('aircraft_lease_cost')        # 13 個月租金總額
aircraft_lease_usage       = Int('aircraft_lease_usage')        # 使用次數
opt.add(aircraft_lease_cost >= 0,  aircraft_lease_cost <= 200)  # 0–2 億
opt.add(aircraft_lease_usage >= 0, aircraft_lease_usage <= 100)

# ——— 公務車租賃 ———
car_lease_cost             = Real('car_lease_cost')             # 12 個月租金總額
car_lease_usage            = Int('car_lease_usage')             # 使用次數
opt.add(car_lease_cost >= 0,  car_lease_cost <= 50)             # 0–5 千萬
opt.add(car_lease_usage >= 0, car_lease_usage <= 100)

# ——— 差旅與交際費 ———
travel_expense_compliant   = Bool('travel_expense_compliant')   # 差旅費用符合內規
entertainment_doc_complete = Bool('entertainment_doc_complete') # 交際費留存完整紀錄

# ——— 授信及控管 ———
credit_eval_performed      = Bool('credit_eval_performed')      # 授信前經濟實質評估
subsidiary_control_effect  = Bool('subsidiary_control_effect')  # 子公司內部控制有效

# -----------------------------------------------------------------------------
# 衍生計算用變數
# -----------------------------------------------------------------------------
# 單位成本（百萬元／次）—避免除以 0
aircraft_cost_per_use = Real('aircraft_cost_per_use')
car_cost_per_use      = Real('car_cost_per_use')
opt.add(
    aircraft_cost_per_use == If(aircraft_lease_usage == 0,
                                aircraft_lease_cost * 1e3,          # 視為極高成本
                                aircraft_lease_cost / aircraft_lease_usage)
)
opt.add(
    car_cost_per_use == If(car_lease_usage == 0,
                           car_lease_cost * 1e3,
                           car_lease_cost / car_lease_usage)
)

# -----------------------------------------------------------------------------
# 法規硬性約束（Hard Constraints）
# -----------------------------------------------------------------------------
# 1) 銀行法 §45-1 ＆ §129 Ⅶ：必須建立並確實執行有效內部控制
opt.add(internal_control_established)           # 已建立
opt.add(internal_control_effective)             # 並且有效運作

# 2) 實施辦法 §12：稽核人員配置與資格
#    ─ 假設每 20 處分支須≥1 名稽核人，且 audit_staff_qualified 為 True
opt.add(auditor_count * 20 >= branch_count)     # 人力足額
opt.add(audit_staff_qualified)                  # 資格符合法規

# 3) 內部控制效果與各細項之邏輯關係（銀行法 §45-1）
opt.add(Implies(internal_control_effective,
                And(
                    credit_eval_performed,
                    travel_expense_compliant,
                    entertainment_doc_complete,
                    subsidiary_control_effect,
                    aircraft_cost_per_use <= 1,    # 單位成本 ≤1 百萬元
                    car_cost_per_use      <= 0.5,  # 單位成本 ≤0.5 百萬元
                )))

# -----------------------------------------------------------------------------
# 邏輯補充：子公司控制應等同母公司要求
opt.add(Implies(subsidiary_control_effect,
                And(
                    aircraft_cost_per_use <= 1,
                    car_cost_per_use      <= 0.5
                )))

# -----------------------------------------------------------------------------
# === START SOFTS CONFIG ===
# -----------------------------------------------------------------------------
# 軟性約束（案例當前數值；違法→UNSAT）
softs = [
    # —— 內部控制核心 ——
    ("銀行法§45-1-已建立制度",                 internal_control_established,   True),
    ("銀行法§45-1-有效運作",                 internal_control_effective,     False),
    ("稽核人員資格符合法規(實施辦法§12)",       audit_staff_qualified,          False),

    # —— 稽核人力 ——
    ("分支機構數",                           branch_count,                   200),
    ("專任稽核人員數",                       auditor_count,                  5),

    # —— 航空器租賃 —— (1.04億/13 月)
    ("航空器租金總額(百萬元)",               aircraft_lease_cost,            104),
    ("航空器使用次數",                       aircraft_lease_usage,           0),

    # —— 公務車租賃 —— (0.984億/12 月)
    ("公務車租金總額(百萬元)",               car_lease_cost,                 9.84),
    ("公務車使用次數",                       car_lease_usage,                2),

    # —— 差旅與交際費 ——
    ("差旅費用符合內規",                     travel_expense_compliant,       False),
    ("交際費留存完整紀錄",                   entertainment_doc_complete,     False),

    # —— 授信及子公司 ——
    ("授信前經濟實質評估已執行",             credit_eval_performed,          False),
    ("子公司內控有效",                       subsidiary_control_effect,      False),

    # —— 額外可調整之重要實數變數 (預設0) ——
    ("航空器單位成本上限(百萬元)",           aircraft_cost_per_use,          0),
    ("公務車單位成本上限(百萬元)",           car_cost_per_use,               0),
]

for label, expr, expected in softs:
    opt.add_soft(expr == expected, weight=1, id=label)

# === END SOFTS CONFIG ===

res = opt.check()
if str(res) == 'sat':
    m = opt.model()
    print("SAT")
    def to_python(val):
        if is_true(val):
            return True
        if is_false(val):
            return False
        try:
            s = val.as_decimal(10)
            if s.endswith('?'):
                s = s[:-1]
            return float(s)
        except:
            return val

    print("\n=== 軟約束比較結果 ===")
    for label, expr, expected in softs:
        actual_z3 = m.eval(expr, model_completion=True)
        actual = to_python(actual_z3)
        print(f"{label}: 預設={expected}, 求解建議值={actual}")
elif str(res) == 'unsat':
    print("UNSAT")
    core = opt.unsat_core()
    print("\n=== Unsat Core (需翻轉的軟約束標籤) ===")
    for c in core:
        print(c)
else:
    print(f"Unknown result: {res}")
