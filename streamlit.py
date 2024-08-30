import streamlit as st
import requests

st.title("4조 프로젝트")

#이미지 업로드 부분 
st.header("이미지를 올려주세요")
uploaded_file = st.file_uploader("이미지 선택", type=["jpg", "jpeg", "png"])
if uploaded_file is not None: 
    image = uploaded_file.read()
    st.image(image, caption="이미지 선택", use_column_width="True")

    # 이미지 전송 API
    if st.button("Predict"):
            files = {"file" : uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8000/predict_image", files=files)
            result = response.json()
            st.write(result)

#Text Generation Section 
st.header("댓글을 기대하세요!")

# 아래 부분은 수정이 좀 필요함(Text 입력을 받지 않기에) 
input_text = st.text_input("Enter some text:")
if st.button("Generate"):
    response = requests.post("http://127.0.0.1:8000/generate_text/", json={"text": input_text})
    result = response.json()
    st.write(result["generated_text"])