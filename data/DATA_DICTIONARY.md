# Data Dictionary - Assessment-Aware Study Planner

| Column Name | Type | Real / Generated | Description / Generation Rule |
|---|---|---|---|
| student_id | Integer | Real | Unique student identifier from OULAD |
| prev_score_1 | Float | Real | Score on assessment 1 (0-100) |
| prev_score_2 | Float | Real | Score on assessment 2 (0-100) |
| prev_score_3 | Float | Real | Score on assessment 3 (0-100) |
| clicks_group_1 | Integer | Real | Clicks in VLE 4 weeks prior to A1 |
| clicks_group_2 | Integer | Real | Clicks in VLE 4 weeks prior to A2 |
| clicks_group_3 | Integer | Real | Clicks in VLE 4 weeks prior to A3 |
| target_mark | Float | Generated | Desired grade, generated as mean(prev_scores) * 1.10 capped at 100 |
| weekly_capacity | Integer | Generated | Free study slots per week per student profile (sampled uniformly from [2, 3, 4]) |
| clicks_per_session | Integer | Generated | Fixed conversion factor (45 clicks = 1 study session) |
