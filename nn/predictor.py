import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MarkPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        # بناء شبكة عصبية صغيرة (Small Neural Network) طبقاً للمتطلبات
        self.model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation='relu',
            max_iter=1000,
            random_state=self.random_state,
            learning_rate_init=0.01
        )
        self.is_trained = False

    def _prepare_features(self, df, extra_sessions=0):
        """
        استخراج الـ 12 Feature المطلوبة:
        Clicks (مقسمة لـ 3 مجموعات + تأثير الجلسات الإضافية), الدرجات السابقة, إلخ.
        """
        X = []
        for _, row in df.iterrows():
            # زيادة النقرات بناءً على الجلسات الإضافية (45 كليك لكل جلسة)
            added_clicks = extra_sessions * row.get('clicks_per_session', 45)
            
            c1 = row['clicks_group_1'] + (added_clicks * 0.2)
            c2 = row['clicks_group_2'] + (added_clicks * 0.3)
            c3 = row['clicks_group_3'] + (added_clicks * 0.5)
            
            p1 = row['prev_score_1']
            p2 = row['prev_score_2']
            p3 = row['prev_score_3']
            
            mean_prev = (p1 + p2 + p3) / 3.0
            total_clicks = c1 + c2 + c3
            
            # الـ 12 Features
            features = [
                c1, c2, c3, total_clicks,
                p1, p2, p3, mean_prev,
                row.get('weekly_capacity', 3),
                row.get('clicks_per_session', 45),
                extra_sessions,
                mean_prev * 0.1 # Feature إضافية تحسب الـ Trend
            ]
            X.append(features)
        return np.array(X)

    def train(self, df):
        X = self._prepare_features(df, extra_sessions=0)
        y = df['target_mark'].values
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # التقييم مقارنة بالـ Baseline (Average Mark Predictor)
        preds_nn = self.predict_raw(X)
        baseline_preds = np.mean(df[['prev_score_1', 'prev_score_2', 'prev_score_3']].values, axis=1)
        
        mse_nn = mean_squared_error(y, preds_nn)
        mse_base = mean_squared_error(y, baseline_preds)
        
        print(f"--- NN Training Completed ---")
        print(f"Neural Network MSE: {mse_nn:.2f} | MAE: {mean_absolute_error(y, preds_nn):.2f}")
        print(f"Baseline (Average) MSE: {mse_base:.2f} | MAE: {mean_absolute_error(y, baseline_preds):.2f}")
        return mse_nn, mse_base

    def predict_raw(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 100.0)

    def predict_student_mark(self, student_row, extra_sessions=0):
        """توقع درجة طالب معين عند إضافة عدد معين من جلسات المذاكرة"""
        if isinstance(student_row, dict):
            student_df = pd.DataFrame([student_row])
        else:
            student_df = pd.DataFrame([student_row.to_dict()])
            
        X = self._prepare_features(student_df, extra_sessions=extra_sessions)
        return float(self.predict_raw(X)[0])

    def test_monotonicity(self, student_row, max_sessions=10):
        """
        Check that predicted marks do not go DOWN when study effort goes up
        (Monotonicity Test Required by Project Deliverables)
        """
        violations = 0
        prev_mark = self.predict_student_mark(student_row, extra_sessions=0)
        
        for s in range(1, max_sessions + 1):
            curr_mark = self.predict_student_mark(student_row, extra_sessions=s)
            if curr_mark < prev_mark - 1e-5: # تباين مسموح صغير جداً
                violations += 1
                print(f"⚠ Monotonicity Violation detected for Student {student_row.get('student_id')}: "
                      f"Sessions {s-1}->{s} dropped mark from {prev_mark:.2f} to {curr_mark:.2f}")
            prev_mark = curr_mark
            
        return violations

if __name__ == '__main__':
    # تجربة سريعة للموديول
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_student_profiles.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        predictor = MarkPredictor()
        predictor.train(df)
        
        # تشغيل اختبار الرتابة على أول طالب
        print("\nRunning Monotonicity Test on Student 1...")
        v = predictor.test_monotonicity(df.iloc[0])
        print(f"Monotonicity Violations Count: {v}")