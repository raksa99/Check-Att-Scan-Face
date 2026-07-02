import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, date

import database
import recognition_engine
import utils
import face_recognition

# Set page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
database.init_db()

# Create photo storage directory
PHOTOS_DIR = "registered_photos"
os.makedirs(PHOTOS_DIR, exist_ok=True)

# Initialize recognition engine in session state
if 'engine' not in st.session_state:
    st.session_state.engine = recognition_engine.FaceRecognitionEngine()
    
# Initialize other session states
if 'latest_log' not in st.session_state:
    st.session_state.latest_log = "No scan detected yet."
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'logged_today' not in st.session_state:
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        today_logs = database.get_attendance_logs(start_date=today_str, end_date=today_str)
        st.session_state.logged_today = {log['user_id'] for log in today_logs}
    except Exception:
        st.session_state.logged_today = set()

# Inject Custom CSS for premium modern dark-glassmorphism aesthetics
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Apply modern typography globally */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
}

/* Main title styling with gradient text */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4d96ff 0%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}

.sub-title {
    font-size: 1.1rem;
    color: #8f9cae;
    font-weight: 400;
    margin-top: 0px;
    margin-bottom: 2rem;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Premium Card / Metric Glassmorphism */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
    backdrop-filter: blur(5px) !important;
    -webkit-backdrop-filter: blur(5px) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stMetric"]:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25) !important;
}

/* Custom styled tabs */
button[data-baseweb="tab"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #8f9cae !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    background-color: transparent !important;
    margin-right: 8px !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background-color: rgba(255, 255, 255, 0.03) !important;
}

button[aria-selected="true"] {
    color: #4D96FF !important;
    background: rgba(77, 150, 255, 0.08) !important;
    border: 1px solid rgba(77, 150, 255, 0.2) !important;
}

/* Button aesthetics styling */
div.stButton > button {
    background: linear-gradient(135deg, #4d96ff 0%, #00cdac 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(77, 150, 255, 0.15) !important;
    width: 100% !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(77, 150, 255, 0.3) !important;
}

div.stButton > button:active {
    transform: translateY(0) !important;
}

/* Primary/Danger Button styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%) !important;
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.15) !important;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(255, 107, 107, 0.3) !important;
}

/* Input Fields styling */
div[data-baseweb="input"], div[data-baseweb="select"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background-color: rgba(255, 255, 255, 0.02) !important;
}

/* Expander border styling */
div[data-testid="stExpander"] {
    border-radius: 12px !important;
    background-color: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
</style>
""", unsafe_allow_html=True)

# Styled Header
st.markdown("""
<div style="text-align: left; margin-bottom: 2rem;">
    <h1 class="main-title">🏫 Smart Attendance System</h1>
    <p class="sub-title">A real-time face recognition dashboard for logging student and employee attendance.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/attendance.png", width=120)
    st.title("System Control Panel")
    
    # Reload engine button
    if st.button("🔄 Reload Face Cache"):
        st.session_state.engine.load_known_faces()
        try:
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_logs = database.get_attendance_logs(start_date=today_str, end_date=today_str)
            st.session_state.logged_today = {log['user_id'] for log in today_logs}
        except Exception:
            pass
        st.success("Face cache reloaded successfully!")
        
    st.markdown("---")
    
    # Database Connection Status
    st.subheader("🌐 Connection Status")
    if database.DB_TYPE == 'SUPABASE':
        st.success("🟢 Connected to Supabase API")
    else:
        st.info("💾 Using Local SQLite database")
        
    st.markdown("---")
    
    # System Stats
    st.subheader("📊 System Stats")
    users = database.get_all_users()
    today_str = datetime.now().strftime('%Y-%m-%d')
    today_logs = database.get_attendance_logs(start_date=today_str, end_date=today_str)
    
    st.metric("Registered Users", len(users))
    st.metric("Check-Ins Today", len(today_logs))
    
    st.markdown("---")
    
    # Database Actions
    st.subheader("⚙️ Database Actions")
    with st.expander("Danger Zone"):
        st.warning("These operations cannot be undone.")
        
        # Clear Logs
        if st.button("🗑️ Clear All Attendance Logs"):
            if database.DB_TYPE == 'SUPABASE':
                try:
                    database.supabase.table("attendance").delete().neq("id", 0).execute()
                    st.toast("Cleared all Supabase attendance logs!", icon="🗑️")
                except Exception as e:
                    st.error(f"Failed to clear logs: {str(e)}")
            else:
                conn = database.get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM attendance")
                conn.commit()
                conn.close()
                st.toast("Cleared all local SQLite attendance logs!", icon="🗑️")
            st.rerun()

# ----------------- TABS SETUP -----------------
tab_scan, tab_directory, tab_reports = st.tabs([
    "🎥 Live Attendance Scanner",
    "👤 User Directory",
    "📊 Reports & Analytics"
])

# ----------------- TAB 1: LIVE ATTENDANCE SCANNER -----------------
with tab_scan:
    st.subheader("Real-Time Face Scanner")
    st.markdown("Turn on the webcam scanner to automatically log daily attendance.")
    
    col_video, col_logs = st.columns([2, 1])
    
    with col_video:
        # Toggle checkbox to start/stop camera loop
        scan_active = st.checkbox("🔌 Enable Attendance Scanner", key="scan_active_toggle")
        
        # Frame placeholder
        frame_placeholder = st.empty()
        
        if scan_active:
            cap = cv2.VideoCapture(0)
            
            # Check camera connection
            if not cap.isOpened():
                st.error("❌ Webcam could not be accessed. Please ensure it's connected and permitted.")
                st.session_state.scan_active_toggle = False
            else:
                # Frame skipping cache variables
                results = []
                frame_count = 0
                
                try:
                    while st.session_state.scan_active_toggle:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to grab frame.")
                            break
                        
                        frame_count += 1
                        
                        # Process face recognition on every 3rd frame to optimize CPU load
                        if frame_count % 3 == 0 or not results:
                            results = st.session_state.engine.recognize_faces_in_frame(frame, tolerance=0.55)
                        
                        for res in results:
                            top, right, bottom, left = res['location']
                            is_known = res['is_known']
                            name = res['name']
                            user_id = res['id']
                            
                            # Choose color: green for known, red for unknown
                            color = (46, 204, 113) if is_known else (231, 76, 60) # BGR: Green vs Red
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                            
                            # Draw label bar
                            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                            cv2.putText(
                                frame, 
                                name, 
                                (left + 6, bottom - 8), 
                                cv2.FONT_HERSHEY_DUPLEX, 
                                0.6, 
                                (255, 255, 255), 
                                1
                            )
                            
                            # Auto log attendance if user is recognized
                            if is_known:
                                # ONLY perform network database write if they haven't checked in yet today in our local memory cache
                                if user_id not in st.session_state.logged_today:
                                    success, msg = database.log_attendance(user_id, name)
                                    if success:
                                        st.session_state.logged_today.add(user_id)
                                        st.session_state.latest_log = msg
                                        st.toast(msg, icon="✅")
                                        # Insert in scan history cache
                                        st.session_state.scan_history.insert(0, {
                                            "time": datetime.now().strftime("%H:%M:%S"),
                                            "name": name,
                                            "id": user_id,
                                            "status": "Logged In"
                                        })
                                        # Limit scan history list size
                                        if len(st.session_state.scan_history) > 10:
                                            st.session_state.scan_history.pop()
                                        
                        # Convert color format BGR (OpenCV) to RGB (Streamlit)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB")
                        
                        # Short sleep to release CPU cycle slightly
                        time.sleep(0.01)
                finally:
                    # Clean up
                    cap.release()
                    frame_placeholder.empty()
                    st.info("Scanner stopped. Camera resource released.")
        else:
            # When scanner is off, show a visual graphic or instructions
            frame_placeholder.info("Click the 'Enable Attendance Scanner' checkbox above to turn on your webcam feed.")
            
    with col_logs:
        st.subheader("🔔 Live Scans & Status")
        
        # Display latest status message
        st.info(f"**Latest Activity:** {st.session_state.latest_log}")
        
        st.markdown("---")
        st.write("⏱️ **Recent Scan History (Session)**")
        
        if st.session_state.scan_history:
            for item in st.session_state.scan_history:
                st.write(f"`{item['time']}` - **{item['name']}** ({item['id']}) - {item['status']}")
        else:
            st.caption("No face recognized yet in this browser session.")

        st.markdown("---")
        st.write("📅 **Today's Checklist**")
        today_data = database.get_attendance_logs(start_date=today_str, end_date=today_str)
        if today_data:
            df_today = pd.DataFrame(today_data)[['name', 'user_id', 'timestamp']]
            df_today.columns = ['Name', 'User ID', 'Log Time']
            st.dataframe(df_today, use_container_width=True)
        else:
            st.caption("No attendance logs recorded for today yet.")


# ----------------- TAB 2: USER DIRECTORY -----------------
with tab_directory:
    st.subheader("Manage User Directory")
    
    col_reg, col_list = st.columns([1, 1])
    
    with col_reg:
        st.markdown("#### ➕ Register New User")
        
        reg_id = st.text_input("User ID (e.g. STU1001, EMP5002)", placeholder="Unique alphanumeric ID").strip()
        reg_name = st.text_input("Full Name", placeholder="First and Last Name").strip()
        
        reg_method = st.radio("Photo Capture Method", ["Webcam Capture", "Upload Image File"])
        
        captured_img = None
        face_encoding = None
        
        if reg_method == "Webcam Capture":
            camera_img = st.camera_input("Take a snapshot of the user's face")
            if camera_img:
                captured_img = camera_img.getvalue()
        else:
            uploaded_file = st.file_uploader("Upload user photo", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                captured_img = uploaded_file.getvalue()
                
        # Dry run face detection if we have an image
        if captured_img:
            # Convert bytes to opencv format
            img_bgr = utils.bytes_to_cv2_image(captured_img)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            try:
                # Extract encoding to test if face is present and valid
                face_encoding = st.session_state.engine.extract_encoding(img_rgb)
                st.success("✅ One face detected and encoded successfully! Ready to register.")
                
                # Show preview with detected face box
                locations = face_recognition.face_locations(img_rgb)
                if locations:
                    top, right, bottom, left = locations[0]
                    preview_img = img_rgb.copy()
                    cv2.rectangle(preview_img, (left, top), (right, bottom), (46, 204, 113), 4)
                    st.image(preview_img, caption="Detected Face Preview", width=250)
            except ValueError as ve:
                st.warning(f"⚠️ Face Validation Failed: {str(ve)}")
                st.info("You can still register this user. Note: they will not be recognized by the real-time scanner.")
                bypass = st.checkbox("🔑 Register anyway (Bypass verification)", key="bypass_validation")
                if bypass:
                    face_encoding = np.zeros(128)
                    st.image(img_rgb, caption="User Photo Preview (Unverified)", width=250)
                else:
                    face_encoding = None
                
        # Register Action
        if st.button("📥 Save & Register User", use_container_width=True):
            if not reg_id:
                st.error("Please enter a User ID.")
            elif not reg_name:
                st.error("Please enter a Full Name.")
            elif captured_img is None:
                st.error("Please capture or upload a photo of the user.")
            elif face_encoding is None:
                st.error("Please capture or upload a valid photo containing exactly one face, or enable 'Bypass verification' above.")
            else:
                # Check if user ID already exists
                existing_users = [u['id'] for u in database.get_all_users()]
                if reg_id in existing_users:
                    st.error(f"User ID '{reg_id}' is already registered. Please choose another ID or delete the existing user first.")
                else:
                    # Save image file
                    photo_filename = f"{reg_id}.jpg"
                    photo_path = os.path.join(PHOTOS_DIR, photo_filename)
                    img_bgr = utils.bytes_to_cv2_image(captured_img)
                    cv2.imwrite(photo_path, img_bgr)
                    
                    # Save user details in database
                    database.add_user(
                        user_id=reg_id,
                        name=reg_name,
                        face_encoding=face_encoding,
                        photo_path=photo_path
                    )
                    
                    # Refresh the recognition cache
                    st.session_state.engine.load_known_faces()
                    st.success(f"Registered user '{reg_name}' successfully!")
                    time.sleep(1)
                    st.rerun()
                    
    with col_list:
        st.markdown("#### 👥 Registered Directory")
        
        all_users = database.get_all_users()
        
        if all_users:
            # Convert list of users to DataFrame
            df_users = pd.DataFrame(all_users)
            # Drop the encoding column for table viewing
            df_users_display = df_users.drop(columns=['face_encoding'])
            df_users_display.columns = ['ID', 'Full Name', 'Photo File Path', 'Registered At']
            
            # Simple search bar
            search_query = st.text_input("🔍 Search Users", placeholder="Filter by Name or ID").strip().lower()
            if search_query:
                df_users_display = df_users_display[
                    df_users_display['ID'].str.lower().str.contains(search_query) |
                    df_users_display['Full Name'].str.lower().str.contains(search_query)
                ]
                
            st.dataframe(df_users_display, use_container_width=True, hide_index=True)
            
            # User Deletion Interface
            st.markdown("#### ❌ Delete User")
            delete_id = st.selectbox(
                "Select User to Delete", 
                options=[u['id'] for u in all_users],
                format_func=lambda uid: f"{uid} - {next((u['name'] for u in all_users if u['id'] == uid), '')}"
            )
            
            if st.button("🗑️ Delete Selected User", type="primary"):
                # Find photo path to delete
                user_obj = next((u for u in all_users if u['id'] == delete_id), None)
                if user_obj and user_obj['photo_path'] and os.path.exists(user_obj['photo_path']):
                    try:
                        os.remove(user_obj['photo_path'])
                    except Exception as e:
                        pass # Ignore photo deleting issues if file was missing
                        
                # Remove from database
                success = database.remove_user(delete_id)
                if success:
                    st.session_state.engine.load_known_faces()
                    st.success(f"Removed user {delete_id} successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to remove user.")
        else:
            st.info("No users have been registered yet. Add one using the form on the left.")


# ----------------- TAB 3: REPORTS & ANALYTICS -----------------
with tab_reports:
    st.subheader("Attendance History & Reports")
    
    # Reports layout
    col_filters, col_stats = st.columns([2, 1])
    
    with col_filters:
        st.markdown("#### 🔍 Filter Options")
        col_date_start, col_date_end = st.columns(2)
        
        with col_date_start:
            # Default to start of current month
            default_start = date(date.today().year, date.today().month, 1)
            start_val = st.date_input("Start Date", value=default_start)
            
        with col_date_end:
            end_val = st.date_input("End Date", value=date.today())
            
        # Optional filter by User
        user_list = database.get_all_users()
        user_options = ["All Users"] + [f"{u['id']} - {u['name']}" for u in user_list]
        selected_user_opt = st.selectbox("Filter by User", user_options)
        
        # Query Logs
        start_str = start_val.strftime('%Y-%m-%d')
        end_str = end_val.strftime('%Y-%m-%d')
        logs = database.get_attendance_logs(start_date=start_str, end_date=end_str)
        
        if logs:
            df_logs = pd.DataFrame(logs)
            
            # Apply user filter if selected
            if selected_user_opt != "All Users":
                sel_uid = selected_user_opt.split(" - ")[0]
                df_logs = df_logs[df_logs['user_id'] == sel_uid]
                
            # Rename columns for presentation
            df_logs_display = df_logs.copy()
            df_logs_display = df_logs_display[['user_id', 'name', 'date', 'timestamp']]
            df_logs_display.columns = ['User ID', 'Name', 'Date', 'Timestamp']
            
            st.markdown(f"**Found {len(df_logs_display)} records:**")
            st.dataframe(df_logs_display, use_container_width=True, hide_index=True)
            
            # Export Options
            st.markdown("#### 📥 Export Data")
            col_csv, col_xlsx = st.columns(2)
            
            with col_csv:
                csv_data = df_logs_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_data,
                    file_name=f"attendance_report_{start_str}_to_{end_str}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            with col_xlsx:
                xlsx_data = utils.convert_df_to_excel(df_logs_display)
                st.download_button(
                    label="📥 Download Excel Report",
                    data=xlsx_data,
                    file_name=f"attendance_report_{start_str}_to_{end_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.info("No attendance records found matching the selected filters.")
            df_logs = None
            
    with col_stats:
        st.markdown("#### 📈 Analytics Summary")
        
        if logs and len(logs) > 0:
            df_logs_stats = pd.DataFrame(logs)
            
            # Metric Cards
            total_logs_count = len(df_logs_stats)
            unique_scans = df_logs_stats['user_id'].nunique()
            reg_users_count = len(database.get_all_users())
            
            # Check-in rate relative to registered users
            attendance_rate = 0.0
            if reg_users_count > 0:
                # average daily active rate or unique scanned / registered
                attendance_rate = (unique_scans / reg_users_count) * 100
                
            st.metric("Total Attendance Logs", total_logs_count)
            st.metric("Unique Checked-in Users", f"{unique_scans} / {reg_users_count}")
            st.metric("Check-in Engagement Rate", f"{attendance_rate:.1f}%")
            
            # Visualization: Check-ins by Date
            st.markdown("##### 📅 Daily Log Volume")
            daily_counts = df_logs_stats.groupby('date').size().reset_index(name='Check-Ins')
            daily_counts = daily_counts.sort_values('date')
            
            st.bar_chart(data=daily_counts, x='date', y='Check-Ins')
        else:
            st.caption("No records available to calculate statistics.")
