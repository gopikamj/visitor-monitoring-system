import firebase_admin
from firebase_admin import credentials, db
import json

cred = credentials.Certificate(r"C:\PRASOBH\COLLEGE\2025last\VotingProject\votingmachine-9b24b-firebase-adminsdk-fbsvc-d04160a907.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://votingmachine-9b24b-default-rtdb.firebaseio.com/"
})

ref = db.reference("/")

# Read entire DB
data = ref.get()

print("----- FULL DATABASE -----")
print(json.dumps(data, indent=4))

# Read specific key
print("\n----- CA VALUE -----")
print("CA:", data.get("CA"))

# Read FAKE_ID
print("\n----- FAKE_ID DATA -----")
fake = data.get("FAKE_ID", {})
for k, v in fake.items():
    print(k, ":", v)
