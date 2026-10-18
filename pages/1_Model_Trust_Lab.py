"""Streamlit page for reproducible blocked-fold ML acceptance testing.

This page reads out-of-fold scores; its displayed predictions are NOT predictions
from the random-split production demonstration model in the main app.
"""
from pathlib import Path
import hashlib
import os
import sqlite3
import json
import pandas as pd
import streamlit as st

from factorymind.trust_lab import (
    METHODS, case_payload, comparison, load_scores, render_hearing,
    request_amd_hearing, select_for_review, trust_evidence_packet,
)
from factorymind.trust_audit import save_trust_review, list_trust_reviews, get_trust_review
from factorymind.bootstrap import ensure_assets

ROOT = Path(__file__).resolve().parents[1]
ensure_assets(ROOT)
SCORES = ROOT / 'artifacts/green_challenge_scored.csv'
METRICS = ROOT / 'artifacts/metrics.json'
REVIEW_DB = os.getenv('FACTORYMIND_REVIEW_DB', str(ROOT / 'data/reviews.sqlite3'))
LABELS = {73: 'Near failure', 80: 'Severe lag', 90: 'Small lag', 100: 'Optimal switching'}

st.set_page_config(page_title='FactoryMind AI · Model Trust Lab', page_icon='🧪', layout='wide')
st.title('🧪 Model Trust Lab')
st.caption('FactoryMind AI — Model Trust Lab · The green light is a prediction, not a guarantee')
st.info('This page audits **blocked out-of-fold** predictions from experimental UCI hydraulic cycles. It does not replay the random-split model from the main app or claim prospective factory safety.')

@st.cache_data
def scores():
    return load_scores(SCORES)

if not SCORES.is_file():
    st.error('Missing stress-test scores. Run: python -m scripts.evaluate_green_challenge')
    st.stop()
try:
    frame = scores()
    scores_sha256 = hashlib.sha256(SCORES.read_bytes()).hexdigest()
except (OSError, ValueError, KeyError) as exc:
    st.error(f'Cannot load audit scores: {type(exc).__name__}: {exc}')
    st.stop()

false_count = int((frame.actual != 100).sum())
metrics = {}
if METRICS.is_file():
    try:
        metrics = json.loads(METRICS.read_text())
    except (ValueError, OSError):
        st.warning('Metrics file is unreadable; the case-level audit remains available.')

st.subheader('1 · The accuracy illusion')
a, b, c, d = st.columns(4)
a.metric('Random holdout accuracy', f"{metrics['accuracy']:.2%}" if 'accuracy' in metrics else 'Not loaded')
b.metric('Blocked accuracy', f"{metrics.get('blocked', {}).get('accuracy', float('nan')):.2%}" if 'accuracy' in metrics.get('blocked', {}) else 'Not loaded')
c.metric('Predicted green (blocked)', f'{len(frame):,}')
d.metric('False-green predictions', f'{false_count:,}')
st.caption('98% random-split accuracy can hide cross-cycle generalization weakness. Neither split measures transfer to a new factory or machine.')

with st.expander('Where are false-green predictions concentrated? · Five held-out blocks'):
    by_fold = (frame.assign(false_green=frame.actual.ne(100).astype(int))
               .groupby('fold', as_index=False)
               .agg(predicted_green=('cycle_id','size'), false_green=('false_green','sum')))
    by_fold['False-green rate'] = by_fold.false_green / by_fold.predicted_green
    st.bar_chart(by_fold.set_index('fold')['False-green rate'], y_label='Error among predicted-green cases', height=190)
    st.dataframe(by_fold, hide_index=True, use_container_width=True)
    st.caption('Retrospective analysis of fixed blocked folds; folds contain different class mixes and experimental regimes. This is not a validated real-factory drift detector.')

st.subheader('2 · Review budget audit')
budget = st.select_slider('Human review capacity — percentage of predicted-green cases per fold', options=[10,20,30], value=20)
results = comparison(frame, budget)
st.bar_chart(results.set_index('Review policy')['False-green caught'], y_label='False-green cases caught', height=270)
display_results = results.copy()
for metric in ('Catch rate', 'Inspection precision'):
    display_results[metric] = display_results[metric] * 100
st.dataframe(display_results, hide_index=True, use_container_width=True,
             column_config={'Catch rate': st.column_config.ProgressColumn('Catch rate', format='%.1f%%', min_value=0, max_value=100),
                            'Inspection precision': st.column_config.ProgressColumn('Inspection precision', format='%.1f%%', min_value=0, max_value=100)})
conf_row = results[results['Review policy'] == 'Confidence-only'].iloc[0]
st.success(f"Confidence-only remains the recommended review policy: {int(conf_row['False-green caught'])} of {false_count} missed degraded cycles flagged by inspecting {int(conf_row['Cycles reviewed'])} of {len(frame)} predicted-green cycles.")
st.warning('The switch metric was explored after observing these same held-out results. Its small advantage at 20% is **not** an independent validation or an approved replacement. The original combined score performed worse.')

st.subheader('3 · Put a prediction on trial')
view = st.selectbox('Choose how to browse cases', ['Curated demonstration (post-hoc examples)', 'Lowest healthy-class score', 'Highest healthy-class score', 'Highest switch fraction', 'Cycle ID'])
curated = [2053, 2184, 507, 638]
if view.startswith('Curated'):
    st.caption('The featured cases were selected **after** analyzing their ground-truth labels. Demonstration only, not an unbiased outcome sample.')
    options = [c for c in curated if c in set(frame.cycle_id)]
else:
    key, ascending = {'Lowest healthy-class score': ('p_optimal_uncalibrated', True),
                      'Highest healthy-class score': ('p_optimal_uncalibrated', False),
                      'Highest switch fraction': ('switch_fraction', False),
                      'Cycle ID': ('cycle_id', True)}[view]
    options = frame.sort_values([key, 'cycle_id'], ascending=[ascending, True], kind='stable').cycle_id.astype(int).tolist()
if not options:
    st.warning('No matching cases in the uploaded scores.')
    st.stop()
cycle = st.selectbox('Operating cycle (blocked fold)', options=options)
record = frame[frame.cycle_id == cycle].iloc[0]
review_ids = select_for_review(frame, 'Confidence-only', budget)
payload = case_payload(record, review_ids)

p1, p2, p3, p4 = st.columns(4)
p1.metric('Original prediction', 'Optimal switching')
p2.metric('Uncalibrated healthy score', f"{payload['healthy_class_score_uncalibrated']:.1%}")
p3.metric('Stress prediction switches', f"{record.switch_fraction:.1%}")
p4.metric('Review by confidence policy', 'YES' if cycle in review_ids else 'NO')
st.caption(f"Blocked fold {int(record.fold)} · {int(record.scenarios_tested)} hypothetical sensor-gain scenarios. No validation label was used for the prediction or review ranking.")

left, right = st.columns([2,1])
with left:
    st.markdown('**Stress-test evidence**')
    comparison_scores = pd.DataFrame({'Healthy-class score': [float(record.base_p_optimal), float(record.min_stressed_p_optimal)]}, index=['Original input','Lowest after gain'])
    st.bar_chart(comparison_scores, height=220, y_label='Uncalibrated score')
    st.caption('Bar chart compares original and minimum scores across scenarios; it does not imply physical degradation or a specific valve failure.')
with right:
    st.markdown('**Most sensitive simulated gain**')
    st.write(f"Sensor: **{record.worst_case_sensor}**")
    st.write(f"Gain: **{record.worst_case_gain_pct:+g}%**")
    st.write(f"Maximum score drop: **{record.max_p_optimal_drop:.1%} points**")
    st.write(f"Prediction switches: **{int(round(float(record.switch_fraction) * int(record.scenarios_tested)))} / {int(record.scenarios_tested)}**")

st.subheader('4 · AMD evidence hearing')
st.caption('Optional live Token Factory call. Dataset ground truth is excluded from the prompt; the AI may only choose bounded evidence and human-review checks.')
if st.button('Ask AMD to prioritize evidence for this case', type='primary'):
    with st.spinner('Requesting bounded AMD investigation…'):
        try:
            report = request_amd_hearing(payload)
            st.session_state['trust_hearing'] = {'cycle': cycle, 'budget': budget, 'scores_sha256': scores_sha256, 'report': report}
        except Exception as exc:
            st.session_state.pop('trust_hearing', None)
            st.error(f'AMD hearing unavailable ({type(exc).__name__}). No generated findings were accepted.')
hearing = st.session_state.get('trust_hearing')
valid_hearing = (hearing and hearing.get('cycle') == cycle and hearing.get('budget') == budget
                 and hearing.get('scores_sha256') == scores_sha256)
if valid_hearing:
    output = render_hearing(hearing['report'], payload)
    st.success(f"{output['provider']} · {output['served_model']} · bounded output accepted")
    st.write(f"Focus: **{output['verdict']}**")
    st.markdown('**Selected, verified evidence**')
    for text in output['evidence']:
        st.write('• ' + text)
    st.markdown('**Suggested human checks**')
    for text in output['checks']:
        st.write('• ' + text)
    st.caption(output['limitations'])

st.subheader('5 · Human decision and evidence passport')
st.caption('This operator decision concerns the **blocked out-of-fold** prediction displayed above. '
           'It is stored separately from main-dashboard approvals for the same cycle. '
           'The held-out dataset label is NOT included in the recorded evidence.')
if valid_hearing:
    with st.form(f'trust_review_{cycle}_{budget}'):
        note = st.text_input('Operator note (optional)', max_chars=2000,
                             placeholder='For example: request verification of sensor traces')
        approve_col, reject_col = st.columns(2)
        approved = approve_col.form_submit_button('Approve inspection review')
        rejected = reject_col.form_submit_button('Reject inspection review')
    if approved or rejected:
        try:
            packet = trust_evidence_packet(record, review_ids, budget, hearing['report'],
                                           scores_file_sha256=scores_sha256)
            review_id = save_trust_review(int(cycle), int(record.fold),
                                         'approve_inspection' if approved else 'reject_inspection',
                                         note, packet, path=REVIEW_DB)
            st.session_state['trust_review_saved'] = {'cycle': cycle, 'id': review_id}
            st.success(f'Trust Lab decision #{review_id} saved with blocked-fold provenance '
                       'and AMD evidence. No equipment was controlled.')
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            st.error(f'Unable to save review ({type(exc).__name__}). No decision recorded.')
else:
    st.info('Run a valid AMD Evidence Hearing for the selected cycle and review budget '
            'before approving or rejecting an inspection.')

with st.expander('Trust Lab audit history for this cycle', expanded=True):
    try:
        history = list_trust_reviews(int(cycle), path=REVIEW_DB)
        if history:
            st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
            review_options = [int(item['id']) for item in history]
            chosen_review = st.selectbox('Inspect saved decision evidence', options=review_options,
                                         key=f'trust_audit_selection_{cycle}')
            saved = get_trust_review(chosen_review, path=REVIEW_DB)
            if saved is not None:
                if saved['evidence_integrity_ok']:
                    st.caption('Stored evidence checksum verified. This checksum alone does not make the log immutable.')
                else:
                    st.error('Saved evidence checksum mismatch; audit data may have changed.')
                st.json(saved['evidence'])
                st.download_button('Download decision passport (JSON)',
                                   data=json.dumps(saved, indent=2, ensure_ascii=False),
                                   file_name=f'factorymind_trust_lab_cycle_{cycle}_decision_{chosen_review}.json',
                                   mime='application/json')
        else:
            st.caption('No Model Trust Lab decisions recorded for this cycle yet.')
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(f'Cannot load Trust Lab review history ({type(exc).__name__}).')

st.divider()
if st.button('Reveal held-out dataset label (evaluation only)'):
    st.session_state['trust_reveal'] = cycle
if st.session_state.get('trust_reveal') == cycle:
    code = int(record.actual)
    if code == 100:
        st.success('Held-out label: Optimal switching — this green prediction was correct.')
    else:
        st.error(f'Held-out label: {LABELS.get(code, str(code))} — the model incorrectly predicted Optimal.')
    st.caption('Dataset label is disclosed after inspection for educational evaluation. No real equipment operation is affected.')
else:
    st.info('The held-out ground truth remains hidden. Decide whether to review the prediction before revealing it.')

st.caption('Research prototype · Blocked out-of-fold model differs from main dashboard model · Experimental UCI test rig, not factory telemetry · AMD Token Factory integration verified by team API probe · No control of equipment.')
