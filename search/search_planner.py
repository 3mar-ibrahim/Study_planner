import os
import sys
import heapq

# إضافة المسار الرئيسي عشان نقدر ننادي الـ NN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from nn.predictor import MarkPredictor

class SearchNode:
    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic=0):
        self.state = state  # Tuple: (A1_sessions, A2_sessions, A3_sessions)
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic = heuristic

    def __lt__(self, other):
        # f(n) = g(n) + h(n)
        return (self.path_cost + self.heuristic) < (other.path_cost + other.heuristic)

def run_search(predictor, student_row, algo='ucs', max_total_sessions=15):
    """
    algo options: 'ucs' (Uniform Cost Search), 'greedy' (Greedy Best-First), 'astar' (A*)
    """
    target = student_row['target_mark']
    initial_state = (0, 0, 0) # 0 sessions for each of the 3 assignments
    
    frontier = []
    nodes_generated = 1
    nodes_expanded = 0
    explored = set()
    
    # دالة بسيطة لحساب الـ Heuristic للـ Greedy و A*
    def get_heuristic(state):
        if algo == 'ucs': 
            return 0
        
        total_s = sum(state)
        pred = predictor.predict_student_mark(student_row, extra_sessions=total_s)
        if pred >= target:
            return 0
        return 1 # Admissible heuristic: We need at least 1 more session
    
    start_node = SearchNode(initial_state, path_cost=0, heuristic=get_heuristic(initial_state))
    
    # Priority Queue (Cost, TieBreaker(generated_id), Node)
    if algo == 'greedy':
        heapq.heappush(frontier, (start_node.heuristic, nodes_generated, start_node))
    else:
        heapq.heappush(frontier, (start_node.path_cost + start_node.heuristic, nodes_generated, start_node))
        
    while frontier:
        _, _, node = heapq.heappop(frontier)
        total_sessions = sum(node.state)
        
        # Test Goal (هل وصلنا للدرجة المستهدفة؟)
        current_pred = predictor.predict_student_mark(student_row, extra_sessions=total_sessions)
        
        if current_pred >= target:
            return extract_solution(node, nodes_generated, nodes_expanded, current_pred)
            
        if node.state in explored or total_sessions >= max_total_sessions:
            continue
            
        explored.add(node.state)
        nodes_expanded += 1
        
        # Actions: إضافة جلسة واحدة لكل من المهام الثلاثة
        for i in range(3):
            new_state = list(node.state)
            new_state[i] += 1
            new_state = tuple(new_state)
            
            if new_state not in explored:
                nodes_generated += 1
                h = get_heuristic(new_state)
                child = SearchNode(new_state, parent=node, action=f"Assign {i+1}", 
                                   path_cost=node.path_cost + 1, heuristic=h)
                
                if algo == 'greedy':
                    heapq.heappush(frontier, (h, nodes_generated, child))
                else: # UCS & A*
                    heapq.heappush(frontier, (child.path_cost + child.heuristic, nodes_generated, child))
                    
    return {"status": "Unreachable", "generated_nodes": nodes_generated, "expanded_nodes": nodes_expanded}

def extract_solution(node, generated, expanded, final_mark):
    plan = []
    curr = node
    while curr.parent:
        plan.append(curr.action)
        curr = curr.parent
    plan.reverse()
    
    return {
        "status": "Success",
        "plan_sequence": plan,
        "allocation": node.state, # (A1, A2, A3)
        "total_sessions": sum(node.state),
        "final_predicted_mark": round(final_mark, 2),
        "generated_nodes": generated,
        "expanded_nodes": expanded
    }

if __name__ == '__main__':
    # اختبار الموديول
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed_student_profiles.csv')
    df = pd.read_csv(data_path)
    
    # تدريب الـ NN الأول
    print("Training NN for Search testing...")
    predictor = MarkPredictor()
    predictor.train(df)
    
    # اختيار طالب لاختبار البحث عليه
    student = df.iloc[0]
    print(f"\n--- Testing Search Algorithms for Student {student['student_id']} ---")
    print(f"Target Mark: {student['target_mark']}")
    
    print("\n1. Uniform Cost Search (UCS):")
    res_ucs = run_search(predictor, student, algo='ucs')
    print(res_ucs)
    
    print("\n2. Greedy Best-First Search:")
    res_greedy = run_search(predictor, student, algo='greedy')
    print(res_greedy)
    
    print("\n3. A* Search:")
    res_astar = run_search(predictor, student, algo='astar')
    print(res_astar)