import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== FACE RECOGNITION ATTENDANCE SYSTEM TEST ===")

# Test 1: Library imports
print("\n[Test 1] Importing ML Libraries...")
try:
    import cv2
    import face_recognition
    print("✅ OpenCV and face_recognition imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import ML Libraries: {e}")
    sys.exit(1)

# Test 2: Database Operations
print("\n[Test 2] Testing Database operations...")
import database
database.DB_TYPE = 'SQLITE'  # Force SQLite mode for testing to protect production data

TEST_DB = "test_attendance.db"

# Cleanup any leftover test db
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)

try:
    # Initialize DB
    database.init_db(TEST_DB)
    print("✅ Database initialized successfully.")

    # Add a user
    dummy_encoding = np.zeros(128)
    database.add_user(
        user_id="TST001",
        name="Test User",
        face_encoding=dummy_encoding,
        photo_path="test_photos/tst001.jpg",
        db_path=TEST_DB
    )
    print("✅ Added user with dummy encoding.")

    # Retrieve users
    users = database.get_all_users(TEST_DB)
    if len(users) == 1 and users[0]['id'] == 'TST001':
        print(f"✅ Retrieved user: {users[0]['name']} (ID: {users[0]['id']})")
        # Check encoding is correct length
        if len(users[0]['face_encoding']) == 128:
            print("✅ Face encoding deserialized correctly as a 128-dimensional list.")
        else:
            print(f"❌ Incorrect encoding dimensions: {len(users[0]['face_encoding'])}")
            sys.exit(1)
    else:
        print(f"❌ Expected 1 user, found {len(users)}")
        sys.exit(1)

    # Log attendance
    success, msg = database.log_attendance("TST001", "Test User", TEST_DB)
    if success:
        print(f"✅ Attendance logged: '{msg}'")
    else:
        print(f"❌ Failed to log initial attendance: {msg}")
        sys.exit(1)

    # Try logging attendance again (should fail due to unique constraint for the day)
    success, msg = database.log_attendance("TST001", "Test User", TEST_DB)
    if not success:
        print(f"✅ Prevented duplicate log: '{msg}'")
    else:
        print(f"❌ Allowed duplicate attendance log on same day!")
        sys.exit(1)

    # Get attendance logs
    logs = database.get_attendance_logs(db_path=TEST_DB)
    if len(logs) == 1:
        print(f"✅ Retrieved attendance log: {logs[0]['name']} checked in on {logs[0]['date']} at {logs[0]['timestamp'].split()[1]}")
    else:
        print(f"❌ Expected 1 attendance log, found {len(logs)}")
        sys.exit(1)

    # Delete user
    deleted = database.remove_user("TST001", TEST_DB)
    if deleted:
        print("✅ User deleted successfully.")
    else:
        print("❌ User deletion failed.")
        sys.exit(1)

finally:
    # Cleanup database
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
        print("🧹 Cleaned up test database file.")

print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! Setup is correct and database constraint functions as expected.")
