# from z3 import *
# from z3.z3util import get_vars
# from collections import defaultdict, deque

# def directed_inference_core(opt: Optimize, root_var: str):
#     # 1. 構建依賴圖 & 記錄每條約束裡所有變數的 sort
#     var_dep_graph = defaultdict(set)
#     var_sorts     = {}           # var_name -> SortRef
#     cons_by_eq    = []
#     others        = []

#     for c in opt.assertions():
#         # --- 先把這條約束裡所有變數的 sort 都記錄 ---
#         for v in get_vars(c):
#             var_sorts[v.decl().name()] = v.sort()

#         # --- 接著判斷是不是等式 ---
#         if is_eq(c):
#             lhs, rhs = c.children()
#             lhs_vars = get_vars(lhs)
#             rhs_vars = get_vars(rhs)

#             # 建立雙向依賴關係
#             for lv in lhs_vars:
#                 lvn = lv.decl().name()
#                 if not rhs_vars:
#                     # 常數賦值 => 自影響
#                     var_dep_graph[lvn].add(lvn)
#                 else:
#                     # 等式左邊變數 -> 右邊變數 (新增雙向依賴)
#                     for rv in rhs_vars:
#                         rvn = rv.decl().name()
#                         var_dep_graph[lvn].add(rvn)  # 左 -> 右
#                         var_dep_graph[rvn].add(lvn)  # 右 -> 左
#                 # 儲存等式，用於後面收核心約束
#                 cons_by_eq.append((lvn, c))
#         else:
#             # 非等式全部歸到 others
#             others.append(c)

#     # 2. 從 root_var 做 BFS，沿著依賴圖找出所有核心變數
#     core_vars = {root_var}
#     q = deque([root_var])
#     while q:
#         v = q.popleft()
#         for nb in var_dep_graph[v]:
#             if nb not in core_vars:
#                 core_vars.add(nb)
#                 q.append(nb)

#     # 3. 收集所有與核心變數相關的約束
#     core_cons = []
#     for lhs_name, c in cons_by_eq:
#         if lhs_name in core_vars:
#             core_cons.append(c)
#     for c in others:
#         if any(v.decl().name() in core_vars for v in get_vars(c)):
#             core_cons.append(c)

#     # 4. 回傳：核心約束 + 只包含核心變數的 sort 字典
#     core_var_domains = {
#         name: var_sorts[name]
#         for name in core_vars
#         if name in var_sorts
#     }
#     return core_cons, core_var_domains

from z3 import *
from z3.z3util import get_vars
from collections import defaultdict, deque

def collect_forward_path(graph, root):
    visited = set()
    q = deque([root])
    while q:
        v = q.popleft()
        for nb in graph[v]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return visited

def directed_inference_core(opt: Optimize, root_var: str, direction='both'):
    """
    direction: 'forward' / 'backward' / 'both'
    - 'forward': 找出受 root_var 影響的變數
    - 'backward': 找出影響 root_var 的來源變數
    - 'both': 預設，與原始版本一致（雙向依賴）
    """

    forward_graph = defaultdict(set)   # A -> B 表 A 會影響 B
    backward_graph = defaultdict(set)  # B -> A 表 B 由 A 推導
    var_sorts = {}                     # 紀錄所有變數的 sort
    cons_by_eq = []                    # (lhs_var_name, constraint)
    others = []

    for c in opt.assertions():
        for v in get_vars(c):
            var_sorts[v.decl().name()] = v.sort()

        if is_eq(c):
            lhs, rhs = c.children()
            lhs_vars = get_vars(lhs)
            rhs_vars = get_vars(rhs)

            for lv in lhs_vars:
                lvn = lv.decl().name()
                if not rhs_vars:
                    # 常數賦值，自依賴
                    forward_graph[lvn].add(lvn)
                    backward_graph[lvn].add(lvn)
                else:
                    for rv in rhs_vars:
                        rvn = rv.decl().name()
                        forward_graph[rvn].add(lvn)   # rv → lv（右邊變數影響左邊變數）
                        backward_graph[lvn].add(rvn)  # lv 依賴 rv
                cons_by_eq.append((lvn, c))
        else:
            others.append(c)

    # 選擇正向或反向依賴圖來走 BFS
    if direction == 'forward':
        graph = forward_graph
    elif direction == 'backward':
        graph = backward_graph
    else:
        # 用 union graph（雙向圖）與原版本一致
        graph = defaultdict(set)
        for k in set(forward_graph) | set(backward_graph):
            graph[k].update(forward_graph[k])
            graph[k].update(backward_graph[k])

    # BFS 擴展核心變數
    core_vars = {root_var}
    q = deque([root_var])
    while q:
        v = q.popleft()
        for nb in graph[v]:
            if nb not in core_vars:
                core_vars.add(nb)
                q.append(nb)

    # 收集與核心變數有關的約束
    reachable_vars = collect_forward_path(forward_graph, root_var)

# 僅收集「這些變數被定義」的 constraint
    core_cons = []
    for lhs_name, c in cons_by_eq:
        if lhs_name in reachable_vars:
            core_cons.append(c)

    # ✅ 僅當該變數是 reachable 且該 constraint 是「單變數邊界」才保留
    for c in others:
        vs = list(get_vars(c))
        if len(vs) == 1:
            var_name = vs[0].decl().name()
            if var_name in reachable_vars:
                core_cons.append(c)

    core_var_domains = {
        name: var_sorts[name]
        for name in core_vars
        if name in var_sorts
    }

    return core_cons, core_var_domains
