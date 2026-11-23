# -*- coding: utf-8 -*-
"""Osmer 네 차원: faith_data.xlsx 학습 → 신앙교육 실습자료_261124.xlsx 실험.
필요: pip install pandas openpyxl scikit-learn matplotlib
VS Code 프로젝트: data/ 폴더의 두 파일을 읽어 results/ 에 저장합니다.
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
TRAIN_FILE = DATA_DIR / 'faith_data.xlsx'
TEST_FILE = DATA_DIR / '신앙교육 실습자료_261124.xlsx'
LABELS = ['신념','관계','헌신','신비']

def make_model():
    return Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2,4), min_df=1)),
        ('perceptron', Perceptron(max_iter=2000, tol=1e-3, random_state=42))
    ])

def split_text(text):
    # 문장부호 뒤에서만 분할. 줄바꿈도 분할. 쉼표는 유지해 의미 훼손 최소화.
    parts = re.split(r'(?<=[.!?])\s+|\n+', str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) >= 4]

train = pd.read_excel(TRAIN_FILE, sheet_name='데이터')
test = pd.read_excel(TEST_FILE, sheet_name='Sheet1')
assert {'신앙진술','차원'} <= set(train.columns)
assert {'조','단계','내용'} <= set(test.columns)
train = train.dropna(subset=['신앙진술','차원']).copy()
assert set(train['차원']) == set(LABELS)

# 학습데이터 내부 5겹 교차검증: 성능 참고용 (실험데이터 성능 아님)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_cv = cross_val_predict(make_model(), train['신앙진술'].astype(str), train['차원'], cv=cv)
print('학습데이터 5-fold 교차검증 정확도:', round(accuracy_score(train['차원'], y_cv), 3))
print(classification_report(train['차원'], y_cv, labels=LABELS, zero_division=0))
cm = pd.DataFrame(confusion_matrix(train['차원'], y_cv, labels=LABELS), index=LABELS, columns=LABELS)

# 전체 100개 진술문으로 최종 모델 학습
model = make_model()
model.fit(train['신앙진술'].astype(str), train['차원'])

# 실험데이터: 9개 기록 -> 문장별 분석 단위
rows = []
for idx, row in test.iterrows():
    sentences = split_text(row['내용'])
    for j, sentence in enumerate(sentences, start=1):
        rows.append({'원본행': idx + 2, '조': (str(row['조']) if str(row['조']).endswith('조') else str(row['조']) + '조'),
                     '단계': str(row['단계']), '문장번호': j, '문장': sentence})
units = pd.DataFrame(rows)
if units.empty:
    raise ValueError('분석할 문장이 없습니다. 실험데이터 내용을 확인하세요.')
units['AI_예측차원'] = model.predict(units['문장'])

# 퍼셉트론 decision_function은 확률이 아닙니다. 불확실성 검토용 점수만 저장.
scores = model.decision_function(units['문장'])
for i, label in enumerate(model.named_steps['perceptron'].classes_):
    units['판별점수_'+label] = scores[:, i]

# 단일 레이블 문장별 비율: 각 조/단계에서 합계 100%
def distribution(data, group_cols):
    counts = pd.crosstab([data[c] for c in group_cols], data['AI_예측차원'])
    counts = counts.reindex(columns=LABELS, fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(1)
    return counts.reset_index(), pct.reset_index()

by_group_n, by_group_pct = distribution(units, ['조'])
by_stage_n, by_stage_pct = distribution(units, ['단계'])
by_both_n, by_both_pct = distribution(units, ['조','단계'])

# 출력 파일
output = RESULTS_DIR / 'faith_osmer_analysis_results.xlsx'
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    units.to_excel(writer, sheet_name='문장별_AI분류', index=False)
    by_group_pct.to_excel(writer, sheet_name='조별_비율', index=False)
    by_group_n.to_excel(writer, sheet_name='조별_빈도', index=False)
    by_stage_pct.to_excel(writer, sheet_name='발달단계별_비율', index=False)
    by_stage_n.to_excel(writer, sheet_name='발달단계별_빈도', index=False)
    by_both_pct.to_excel(writer, sheet_name='조_단계별_비율', index=False)
    by_both_n.to_excel(writer, sheet_name='조_단계별_빈도', index=False)
    cm.to_excel(writer, sheet_name='학습교차검증_혼동행렬')
print('\n분석 문장 수:', len(units))
print('\n조별 비율(%)\n', by_group_pct.to_string(index=False))
print('\n발달 단계별 비율(%)\n', by_stage_pct.to_string(index=False))
print('\n결과 파일:', output)

# 영어 범례로 Colab 한글 글꼴 오류 방지
plot_df = by_group_pct.set_index('조')[LABELS].rename(columns={
    '신념':'Belief','관계':'Relationship','헌신':'Commitment','신비':'Mystery'})
plot_df.index = ['Group ' + str(v).replace('조', '') for v in plot_df.index]
ax = plot_df.plot(kind='bar', stacked=True, figsize=(8,5), ylim=(0,100))
ax.set_title('Osmer Dimensions by Group (Perceptron Predictions)')
ax.set_ylabel('Percent of classified text units')
ax.set_xlabel('Group')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'faith_osmer_group_chart.png', dpi=160)
plt.close()

