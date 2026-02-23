import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# --- 1. การตั้งค่าหน้าเว็บ (Frontend Configuration) ---
st.set_page_config(
    page_title="TERRA - AI Fertilizer System",
    page_icon="🌱",
    layout="wide"
)

# --- 2. การเชื่อมต่อ Firebase (Backend - Cloud Connection) ---
@st.cache_resource
def init_firebase():
    # ตรวจสอบว่ามีการเชื่อมต่ออยู่แล้วหรือไม่ เพื่อป้องกันการเชื่อมต่อซ้ำ
    if not firebase_admin._apps:
        # ดึงข้อมูลกุญแจจาก Streamlit Secrets เพื่อความปลอดภัย
        key_dict = st.secrets["firebase_key"]
        cred = credentials.Certificate(dict(key_dict))
        firebase_admin.initialize_app(cred)
    return firestore.client()

# --- 3. การโหลดโมเดล AI (AI Brain Loading) ---
@st.cache_resource
def load_terra_model():
    # โหลดไฟล์สมอง AI ที่คุณอัปโหลดไว้บน GitHub
    return joblib.load("terra_model.pkl")

# เรียกใช้งานฟังก์ชันที่สร้างไว้
db = init_firebase()
model_data = load_terra_model()
clf = model_data['classifier']
reg = model_data['regressor']

# --- 4. ฟังก์ชันดึงข้อมูลสดจาก Cloud ---
def get_sensor_latest():
    # ดึงข้อมูลจาก Collection 'sensor_data' และ Document 'latest'
    doc_ref = db.collection('sensor_data').document('latest')
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None

# --- 5. ส่วนแสดงผลหน้าเว็บ (UI Design) ---
st.title("🌱 TERRA: ระบบแนะนำปุ๋ยลำไยอัจฉริยะ")
st.markdown("วิเคราะห์ธาตุอาหารในดินและแนะนำการใส่ปุ๋ยด้วย AI โดยกลุ่ม Computer Engineering CMU")

# ดึงค่าล่าสุดมาโชว์
sensor_data = get_sensor_latest()

if sensor_data:
    # แสดงค่าจาก Sensor เป็นกล่อง Metric สวยๆ
    st.subheader("📡 ข้อมูลสดจากเซนเซอร์ในสวน (Real-time Cloud Data)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Nitrogen (N)", f"{sensor_data.get('N', 0)} g")
    m2.metric("Phosphorus (P)", f"{sensor_data.get('P', 0)} g")
    m3.metric("Potassium (K)", f"{sensor_data.get('K', 0)} g")
    m4.metric("ค่า pH (ดิน)", sensor_data.get('pH', 0))
    m5.metric("ความชื้น (Moisture)", f"{sensor_data.get('Moist', 0)}%")
    
    st.divider()

    # ส่วนรับข้อมูลจาก User
    st.subheader("⚙️ ตั้งค่าการประมวลผล (User Input)")
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        stage_name = st.selectbox(
            "ระยะการเจริญเติบโตของลำไย:",
            ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"]
        )
    with col_input2:
        yield_target = st.number_input("เป้าหมายผลผลิตที่ต้องการ (กก./ต้น):", min_value=1, value=100)

    # ปุ่มสั่งงาน AI
    if st.button("🚀 เริ่มวิเคราะห์แผนการใส่ปุ๋ย", use_container_width=True):
        # แปลงชื่อระยะเป็นตัวเลขตามที่ AI เคยเรียนรู้
        stage_map = {"ฟื้นต้น": 1, "สะสมอาหาร": 2, "ขยายผล": 3, "ก่อนเก็บเกี่ยว": 4}
        
        # จัดเตรียมข้อมูลเข้า AI (ต้องเรียง Column ให้เหมือนตอนเทรนใน Colab)
        input_df = pd.DataFrame([[
            sensor_data.get('N', 0), sensor_data.get('P', 0), sensor_data.get('K', 0),
            sensor_data.get('pH', 0), sensor_data.get('Moist', 0), 
            stage_map[stage_name], yield_target
        ]], columns=['N_soil', 'P_soil', 'K_soil', 'pH', 'Moisture', 'Stage', 'Target_Yield_kg'])

        # ให้ AI ทายผล
        action_result = clf.predict(input_df)[0]
        nums_result = reg.predict(input_df)[0] # [Lime, N, P, K]

        # แสดงผลลัพธ์
        st.success(f"### 💡 ผลวิเคราะห์จาก AI: \n {action_result}")
        
        st.markdown("#### 🧪 ปริมาณที่ต้องเติมโดยประมาณ:")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.info(f"**ไนโตรเจน (N)**\n{nums_result[1]:.1f} กรัม")
        res_col2.info(f"**ฟอสฟอรัส (P)**\n{nums_result[2]:.1f} กรัม")
        res_col3.info(f"**โพแทสเซียม (K)**\n{nums_result[3]:.1f} กรัม")
        res_col4.warning(f"**ปูนขาว (Lime)**\n{nums_result[0]:.2f} กิโลกรัม")

else:
    st.error("❌ ไม่พบข้อมูลเซนเซอร์ในระบบ Cloud (กรุณาตรวจสอบการส่งค่าจาก Arduino/ESP32)")
    if st.button("🔄 ลองดึงข้อมูลอีกครั้ง"):
        st.rerun()

# ฟุตเตอร์
st.divider()
st.caption("Project Terra | Computer Engineering, Chiang Mai University 2026")