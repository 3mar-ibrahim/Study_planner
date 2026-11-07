import copy

class CSPScheduler:
    def __init__(self, allocation, weekly_capacity, weeks=12):
        self.allocation = allocation # e.g. (5, 0, 0)
        self.weekly_capacity = weekly_capacity
        self.total_slots = weeks * weekly_capacity
        
        # تحديد ميعاد تسليم كل مادة (Deadlines) - كافتراض منطقي
        # A1 in week 4, A2 in week 8, A3 in week 12
        self.deadlines = {
            1: 4 * weekly_capacity - 1,
            2: 8 * weekly_capacity - 1,
            3: 12 * weekly_capacity - 1
        }
        
        self.variables = []
        self.domains = {}
        
        # إنشاء المتغيرات (Variables) وتحديد نطاقاتها (Domains)
        for assign_idx, count in enumerate(allocation):
            a_id = assign_idx + 1
            for session_num in range(1, count + 1):
                var_name = f"A{a_id}_S{session_num}"
                self.variables.append(var_name)
                # Domain: كل الأوقات المتاحة من 0 لحد ميعاد التسليم
                self.domains[var_name] = list(range(self.deadlines[a_id] + 1))
                
        # العدادات المطلوبة (Counters)
        self.assignments_count = 0
        self.backtracks_count = 0

    def is_consistent(self, var, val, assignment):
        """التحقق من القيود (Constraints)"""
        # 1. No Overlap: لا يمكن لجلسة أخرى أن تأخذ نفس الوقت
        if val in assignment.values():
            return False
            
        # 2. Ordering: الجلسة رقم 2 يجب أن تكون بعد الجلسة رقم 1 لنفس المادة
        a_id, s_id = var.split('_')
        s_num = int(s_id[1:])
        
        for other_var, other_val in assignment.items():
            o_a_id, o_s_id = other_var.split('_')
            if o_a_id == a_id:
                o_s_num = int(o_s_id[1:])
                if s_num > o_s_num and val <= other_val:
                    return False
                if s_num < o_s_num and val >= other_val:
                    return False
        return True

    def ac3(self):
        """Arc Consistency (AC-3) - تقليم الأوقات المستحيلة مبكراً"""
        queue = [(xi, xj) for xi in self.variables for xj in self.variables if xi != xj]
        
        while queue:
            xi, xj = queue.pop(0)
            if self.revise(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False # مستحيل الحل
                for xk in self.variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))
        return True

    def revise(self, xi, xj):
        revised = False
        domain_xi = self.domains[xi].copy()
        
        for x in domain_xi:
            # نبحث عن قيمة متوافقة في xj
            conflict = True
            for y in self.domains[xj]:
                if x != y: # Constraint مبسط: فقط منع التداخل
                    # لو كانوا نفس المادة، نطبق قيد الترتيب أيضاً
                    if xi.split('_')[0] == xj.split('_')[0]:
                        s_xi = int(xi.split('_')[1][1:])
                        s_xj = int(xj.split('_')[1][1:])
                        if (s_xi < s_xj and x < y) or (s_xi > s_xj and x > y):
                            conflict = False
                            break
                    else:
                        conflict = False
                        break
            
            if conflict:
                self.domains[xi].remove(x)
                revised = True
        return revised

    def backtrack(self, assignment, inference='none'):
        """Backtracking Search مع دعم الـ Forward Checking"""
        if len(assignment) == len(self.variables):
            return assignment
            
        # اختيار المتغير التالي الذي لم يتم جدولته
        unassigned = [v for v in self.variables if v not in assignment]
        var = unassigned[0]
        
        for val in self.domains[var]:
            if self.is_consistent(var, val, assignment):
                assignment[var] = val
                self.assignments_count += 1
                
                # Forward Checking (FC)
                local_domains = copy.deepcopy(self.domains)
                fc_failed = False
                
                if inference in ['fc', 'ac3']:
                    # حذف الوقت المحجوز من باقي المتغيرات
                    for unassigned_var in unassigned[1:]:
                        if val in self.domains[unassigned_var]:
                            self.domains[unassigned_var].remove(val)
                        if len(self.domains[unassigned_var]) == 0:
                            fc_failed = True
                            break
                
                if not fc_failed:
                    result = self.backtrack(assignment, inference)
                    if result is not None:
                        return result
                        
                # Backtrack
                del assignment[var]
                self.domains = local_domains # إرجاع النطاقات
                self.backtracks_count += 1
                
        return None

    def solve(self, method='bt'):
        """
        طرق الحل: 'bt' (Backtracking), 'fc' (Forward Checking), 'ac3' (AC-3 + FC)
        """
        self.assignments_count = 0
        self.backtracks_count = 0
        
        # No sessions to place → already a valid (empty) schedule
        if not self.variables:
            return {
                "status": "Success",
                "schedule": {},
                "assignments": 0,
                "backtracks": 0,
            }

        if method == 'ac3':
            if not self.ac3():
                return {"status": "Failed by AC-3 pre-processing"}
                
        assignment = self.backtrack({}, inference=method)
        
        # Use `is not None` so an empty schedule {} (0 sessions) counts as success
        if assignment is not None:
            return {
                "status": "Success",
                "schedule": assignment,
                "assignments": self.assignments_count,
                "backtracks": self.backtracks_count
            }
        return {"status": "Failed to schedule", "backtracks": self.backtracks_count}

if __name__ == '__main__':
    # اختبار الـ CSP على نفس النتيجة التي خرجت من الـ Search
    allocation = (5, 0, 0) # 5 جلسات للمادة الأولى
    weekly_capacity = 3    # 3 أوقات فاضية في الأسبوع
    
    print(f"--- Testing CSP Scheduler ---")
    print(f"To Schedule: {allocation} with capacity {weekly_capacity}/week\n")
    
    # 1. Backtracking Only (BT)
    scheduler_bt = CSPScheduler(allocation, weekly_capacity)
    res_bt = scheduler_bt.solve(method='none')
    print("1. Standard Backtracking (BT):")
    print(res_bt)
    
    # 2. Backtracking + Forward Checking (BT+FC)
    scheduler_fc = CSPScheduler(allocation, weekly_capacity)
    res_fc = scheduler_fc.solve(method='fc')
    print("\n2. Backtracking + Forward Checking (BT+FC):")
    print(res_fc)
    
    # 3. AC-3
    scheduler_ac3 = CSPScheduler(allocation, weekly_capacity)
    res_ac3 = scheduler_ac3.solve(method='ac3')
    print("\n3. Backtracking + FC + AC-3:")
    print(res_ac3)