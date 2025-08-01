
from z3 import *
# -----------------------------------------------------------------------------
# 建立 Z3 Optimize 物件（支援軟性約束）
opt = Optimize()

# -----------------------------------------------------------------------------
# 變數定義與上下界設定
# -----------------------------------------------------------------------------
pre_self_capital = Real('pre_self_capital')  # 改善前自有資本（百萬元）
opt.add(pre_self_capital >= 0, pre_self_capital <= 1000)

risk_capital = Real('risk_capital')  # 風險資本（百萬元）
opt.add(risk_capital > 0, risk_capital <= 1000)

added_capital = Real('added_capital')  # 改善計畫中新增資本（百萬元）
opt.add(added_capital >= 0, added_capital <= 500)

net_worth_ratio = Real('net_worth_ratio')  # 112年底淨值比率（%）
opt.add(net_worth_ratio >= 0, net_worth_ratio <= 10)

# -----------------------------------------------------------------------------
# 計算公式（硬性約束）
# -----------------------------------------------------------------------------
pre_CAR = Real('pre_CAR')  # 改善前資本適足率（%）
opt.add(pre_CAR == (pre_self_capital / risk_capital) * 100)

final_CAR = Real('final_CAR')  # 改善後資本適足率（%）
opt.add(final_CAR == ((pre_self_capital + added_capital) / risk_capital) * 100)

improvement_plan = Real('improvement_plan')  # 提升幅度（%）
opt.add(improvement_plan == final_CAR - pre_CAR)

# -----------------------------------------------------------------------------
# 細分改善計畫內部結構
# -----------------------------------------------------------------------------
added_capital_base = Real('added_capital_base')     # 基本資本新增
added_capital_private = Real('added_capital_private')  # 私募資本新增
opt.add(added_capital == added_capital_base + added_capital_private)
opt.add(added_capital_base >= 0, added_capital_base <= 500)
opt.add(added_capital_private >= 0, added_capital_private <= 500)

base_improvement = Real('base_improvement')      # 基本部分提升
private_improvement = Real('private_improvement')  # 私募部分提升
opt.add(base_improvement == (added_capital_base / risk_capital) * 100)
opt.add(private_improvement == (added_capital_private / risk_capital) * 100)
opt.add(improvement_plan == base_improvement + private_improvement)

# -----------------------------------------------------------------------------
# 定義「資本等級」
# 1: 資本適足 (CAR>=200 且 淨值比率>=3)
# 2: 資本不足 (CAR>=150)
# 3: 資本顯著不足 (CAR>=50)
# 4: 資本嚴重不足
# -----------------------------------------------------------------------------
pre_capital_grade = Int('pre_capital_grade')
opt.add(pre_capital_grade == If(And(pre_CAR >= 200, net_worth_ratio >= 3), 1,
                          If(pre_CAR >= 150, 2,
                          If(pre_CAR >= 50, 3, 4))))

capital_grade = Int('capital_grade')
opt.add(capital_grade == If(And(final_CAR >= 200, net_worth_ratio >= 3), 1,
                      If(final_CAR >= 150, 2,
                      If(final_CAR >= 50, 3, 4))))

# -----------------------------------------------------------------------------
# 改善計畫執行布林變數（易讀註解）
# has_signed_contract: True = 已簽訂私募負債型特別股合約；False = 尚未簽約
# has_schedule_detail: True = 已有具體簽約時程及程序；False = 無具體時程
# has_complete_plan: True = 已提交完整且可行的增資/財務/業務改善計畫；False = 無完整計畫
# -----------------------------------------------------------------------------
has_signed_contract = Bool('has_signed_contract')
has_schedule_detail = Bool('has_schedule_detail')
has_complete_plan = Bool('has_complete_plan')

# -----------------------------------------------------------------------------
# 監理措施布林變數（True = 採取該措施；False = 未採取）
# 第143-6第1款：資本不足措施
stop_new_products = Bool('stop_new_products')            # 停止商品銷售
restrict_fund_use = Bool('restrict_fund_use')            # 限制資金運用
limit_remuneration = Bool('limit_remuneration')          # 限制給付酬勞
other_measures = Bool('other_measures')                  # 其他必要處置

# 第143-6第2款：資本顯著不足額外措施
remove_officer = Bool('remove_officer')                  # 解除負責人職務
suspend_officer = Bool('suspend_officer')                # 停止負責人執行職務
require_asset_approval = Bool('require_asset_approval')  # 資產取得/處分須核准
dispose_specific_assets = Bool('dispose_specific_assets')# 處分特定資產
restrict_related_transactions = Bool('restrict_related_transactions')  # 限制關係人交易
reduce_officer_pay = Bool('reduce_officer_pay')          # 降低負責人報酬
limit_branches = Bool('limit_branches')                  # 限制分支機構
other_significant = Bool('other_significant')            # 其他必要處置

# 第143-6第3款：資本嚴重不足特別處分
enact_article149 = Bool('enact_article149')              # 採行第149-3-1處分

# -----------------------------------------------------------------------------
# 法規硬性要求
# -----------------------------------------------------------------------------
opt.add(final_CAR >= 200)       # 最終CAR必須≥200%
opt.add(net_worth_ratio >= 3)   # 淨值比率必須≥3%

# 達資本適足 (等級==1)：三項計畫指標皆須為 True
opt.add(Implies(capital_grade == 1,
                And(has_signed_contract,
                    has_schedule_detail,
                    has_complete_plan)))

# 資本不足 (等級==2)：至少採取一項第1款措施
opt.add(Implies(capital_grade == 2,
                Or(stop_new_products,
                   restrict_fund_use,
                   limit_remuneration,
                   other_measures)))

# 資本顯著不足 (等級==3)：須採取第1款和第2款至少一項
opt.add(Implies(capital_grade == 3,
                And(
                    Or(stop_new_products,
                       restrict_fund_use,
                       limit_remuneration,
                       other_measures),
                    Or(remove_officer,
                       suspend_officer,
                       require_asset_approval,
                       dispose_specific_assets,
                       restrict_related_transactions,
                       reduce_officer_pay,
                       limit_branches,
                       other_significant)
                )))

# 資本嚴重不足 (等級==4)：採取第1、2款及第149-3-1處分
opt.add(Implies(capital_grade == 4,
                And(
                    Or(stop_new_products,
                       restrict_fund_use,
                       limit_remuneration,
                       other_measures),
                    Or(remove_officer,
                       suspend_officer,
                       require_asset_approval,
                       dispose_specific_assets,
                       restrict_related_transactions,
                       reduce_officer_pay,
                       limit_branches,
                       other_significant),
                    enact_article149
                )))

is_life_insurer = Bool('is_life_insurer')  # True=人身保險業；False=財產保險業

# 風險資本細項
asset_risk            = Real('asset_risk')            # 資產風險
insurance_risk        = Real('insurance_risk')        # 保險風險（人身保險）
interest_rate_risk    = Real('interest_rate_risk')    # 利率風險（人身保險）
other_risk            = Real('other_risk')            # 其他風險

credit_risk           = Real('credit_risk')           # 信用風險（財產保險）
underwriting_risk     = Real('underwriting_risk')     # 核保風險（財產保險）
alm_risk              = Real('alm_risk')              # 資產負債配置風險（財產保險）

# 自有資本分類
tier1_unrestricted    = Real('tier1_unrestricted')    # 第一類非限制性資本
tier1_restricted      = Real('tier1_restricted')      # 第一類限制性資本
tier2_capital         = Real('tier2_capital')         # 第二類資本

# -----------------------------------------------------------------------------
# 邊界條件
for v in [asset_risk, insurance_risk, interest_rate_risk, other_risk,
          credit_risk, underwriting_risk, alm_risk]:
    opt.add(v >= 0)

for v in [tier1_unrestricted, tier1_restricted, tier2_capital]:
    opt.add(v >= 0)

# -----------------------------------------------------------------------------
# 根據法規 3 條，風險資本等於各項風險之加總
opt.add(
    If(is_life_insurer,
       # 人身保險業：4 項
       risk_capital == asset_risk
                       + insurance_risk
                       + interest_rate_risk
                       + other_risk,
       # 財產保險業：5 項
       risk_capital == asset_risk
                       + credit_risk
                       + underwriting_risk
                       + alm_risk
                       + other_risk
    )
)

# -----------------------------------------------------------------------------
# 根據法規 2 條，自有資本等於三類資本之和
opt.add(pre_self_capital == tier1_unrestricted
                            + tier1_restricted
                            + tier2_capital)
# === START SOFTS CONFIG ===
# -----------------------------------------------------------------------------
# 軟性約束（保留案例檢視）
# -----------------------------------------------------------------------------
softs = [
    # 原有案例检视
    ("112年提升過後之資本適足率",         pre_CAR,               150),
    ("年度計畫提升幅度",               improvement_plan,      30.4),
    ("112年底淨值比率",               net_worth_ratio,       2.97),
    ("基本提升比率",                   base_improvement,      17.9),
    ("私募改善提升比率",               private_improvement,   12.5),
    ("自有資本",                       pre_self_capital,      150),
    ("風險資本",                       risk_capital,          100),
    ("最終CAR",                       final_CAR,             pre_CAR),
    ("資本等級",                       capital_grade,         pre_capital_grade),
    ("已簽訂私募負債型特別股合約(False=未簽, True=已簽)", has_signed_contract, False),
    ("具體簽約時程及程序 (False=無, True=有)",           has_schedule_detail,  False),
    ("提交完整且可行的增資/財務/業務改善計畫 (False=無, True=有)", has_complete_plan, False),

    # 法规第3条─人身保险业 风险项目（预设 0）
    ("人身保險業-資產風險 (保資第3條第1款(一))",         asset_risk,         0),
    ("人身保險業-保險風險 (保資第3條第1款(二))",         insurance_risk,     0),
    ("人身保險業-利率風險 (保資第3條第1款(三))",         interest_rate_risk, 0),
    ("人身保險業-其他風險 (保資第3條第1款(四))",         other_risk,         0),

    # 法规第3条─财产保险业 风险项目（预设 0）
    ("財產保險業-資產風險 (保資第3條第2款(一))",         asset_risk,         0),
    ("財產保險業-信用風險 (保資第3條第2款(二))",         credit_risk,        0),
    ("財產保險業-核保風險 (保資第3條第2款(三))",         underwriting_risk,  0),
    ("財產保險業-資產負債配置風險 (保資第3條第2款(四))",  alm_risk,           0),
    ("財產保險業-其他風險 (保資第3條第2款(五))",         other_risk,         0),

    # 法规第2条─自有资本三大类（预设 0）
    ("第一類非限制性資本 (保資第2條)",                    tier1_unrestricted, 0),
    ("第一類限制性資本 (保資第2條)",                      tier1_restricted,   0),
    ("第二類資本 (保資第2條)",                            tier2_capital,      0),
]

# 一次性加入所有软性约束
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
