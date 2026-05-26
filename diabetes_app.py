# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="당뇨병 예측 시스템",
    page_icon="🩺",
    layout="wide"
)

# =========================
# CSS 스타일
# =========================
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

/* 제목 박스 */
.title-box {
    background: linear-gradient(135deg, #4F8BF9, #6A5AE0);
    padding: 35px;
    border-radius: 25px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

/* 버튼 */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #4F8BF9, #6A5AE0);
    color: white;
    border-radius: 15px;
    border: none;
    padding: 14px;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg, #3b73db, #5948d6);
    color: white;
}

/* 슬라이더 색상 */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #4F8BF9, #6A5AE0);
}

/* 슬라이더 손잡이 */
.stSlider [role="slider"] {
    background-color: white;
    border: 3px solid #6A5AE0;
}

/* 결과 카드 */
.result-good {
    background-color: #d4edda;
    color: #155724;
    padding: 25px;
    border-radius: 20px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.result-bad {
    background-color: #f8d7da;
    color: #721c24;
    padding: 25px;
    border-radius: 20px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# =========================
# 모델 불러오기
# =========================
scaler = joblib.load("diabetes_scaler.pkl")
log_model_eng = joblib.load("diabetes_model.pkl")

# =========================
# 제목
# =========================
st.markdown("""
<div class="title-box">
    <h1>🩺 AI 당뇨병 예측 시스템</h1>
    <p style="font-size:18px;">
        건강 데이터를 입력하면 AI가 당뇨 위험도를 예측합니다.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# 입력 UI
# =========================
col1, col2 = st.columns(2)

with col1:

    st.markdown("## 📋 기본 정보")

    # 정수 슬라이더
    preg = st.slider(
        "임신 횟수",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.slider(
        "포도당 수치",
        min_value=0,
        max_value=250,
        value=120,
        step=1
    )

    bp = st.slider(
        "혈압",
        min_value=0,
        max_value=150,
        value=70,
        step=1
    )

    skin = st.slider(
        "피부 두께",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

with col2:

    st.markdown("## 🧪 건강 정보")

    insulin = st.slider(
        "인슐린 수치",
        min_value=0,
        max_value=900,
        value=79,
        step=1
    )

    bmi = st.slider(
        "체질량지수 (BMI)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    dpf = st.slider(
        "당뇨병 계층 함수",
        min_value=0.0,
        max_value=3.0,
        value=0.627,
        step=0.001
    )

    age = st.slider(
        "나이",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

# =========================
# 예측 버튼
# =========================
if st.button("🔍 당뇨병 예측하기"):

    # 데이터프레임 생성
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            '임신횟수',
            '포도당',
            '혈압',
            '피부두께',
            '인슐린',
            '체질량지수',
            '당뇨병계층함수',
            '나이'
        ]
    )

    # 파생 변수 생성
    input_data['고포도당_위험'] = (
        input_data['포도당'] >= 140
    ).astype(int)

    # 컬럼 순서 맞추기
    input_data = input_data[
        [
            '임신횟수',
            '포도당',
            '혈압',
            '피부두께',
            '인슐린',
            '체질량지수',
            '당뇨병계층함수',
            '나이',
            '고포도당_위험'
        ]
    ]

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    predicted = log_model_eng.predict(input_scaled)
    prob = log_model_eng.predict_proba(input_scaled)

    diabetes_prob = prob[0][1] * 100

    st.markdown("---")

    # =========================
    # 결과 출력
    # =========================
    if predicted[0] == 1:

        st.markdown(f"""
        <div class="result-bad">
            ⚠️ 당뇨병 위험이 있습니다.<br><br>
            위험 확률: {diabetes_prob:.1f}%
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-good">
            ✅ 정상 범위로 예측됩니다.<br><br>
            당뇨 위험 확률: {diabetes_prob:.1f}%
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # 진행 바
    # =========================
    st.markdown("## 📈 당뇨 위험도")

    st.progress(min(int(diabetes_prob), 100))

    st.metric(
        label="당뇨 위험 확률",
        value=f"{diabetes_prob:.1f}%"
    )

    # =========================
    # 입력 데이터 표시
    # =========================
    st.markdown("## 📊 입력 데이터")

    st.dataframe(
        input_data,
        use_container_width=True
    )

# =========================
# 하단 안내
# =========================
st.markdown("---")

st.caption(
    "※ 본 서비스는 AI 기반 예측 보조 도구이며 실제 의료 진단을 대체하지 않습니다."
)

#py -m streamlit run diabetes_app.py 
#py -m pip install streamlit
#py -m pip install joblib
#py -m pip install scikit-learn