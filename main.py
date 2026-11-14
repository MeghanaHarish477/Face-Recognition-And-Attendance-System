import cv2
from datetime import datetime
import os
import customtkinter as ctk 
from tkinter import messagebox

ctk.set_appearance_mode("Light") 
ctk.set_default_color_theme("blue") 

recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists('trainer.yml'):
    recognizer.read('trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
names = ['MEGHANA', 'HARISH', 'SANGEETHA']

if not os.path.exists("Attendance.csv"):
    with open("Attendance.csv", "w") as f:
        f.write("Name,Time\n")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Smart Attendance System")
        self.geometry("500x550")

        self.header_label = ctk.CTkLabel(self, text="BIOMETRIC PORTAL", font=ctk.CTkFont(size=24, weight="bold"))
        self.header_label.pack(pady=(40, 10))

        self.status_frame = ctk.CTkFrame(self, width=400, height=60)
        self.status_frame.pack(pady=10)
        self.status_label = ctk.CTkLabel(self.status_frame, text="System Ready", text_color="#2da44e", font=ctk.CTkFont(weight="bold"))
        self.status_label.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkButton(self, text="START SCANNER", command=self.start_attendance, fg_color="#2da44e", height=45, width=250).pack(pady=15)
        ctk.CTkButton(self, text="VIEW LOGS", command=self.view_attendance, fg_color="#0969da", height=45, width=250).pack(pady=15)
        ctk.CTkButton(self, text="EXIT SYSTEM", command=self.quit, fg_color="#cf222e", height=45, width=250).pack(pady=15)

    def start_attendance(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if not ret: break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                name = names[id] if confidence < 80 else "UNKNOWN"
                if name != "UNKNOWN":
                    time_now = datetime.now().strftime('%H:%M:%S')
                    with open("Attendance.csv", "r+") as f:
                        if name not in f.read(): f.writelines(f'\n{name},{time_now}')
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Attendance Feed - Press ESC to Quit", img)
            if cv2.waitKey(1) == 27: break
        cap.release()
        cv2.destroyAllWindows()

    def view_attendance(self):
        if os.path.exists("Attendance.csv"):
            with open("Attendance.csv", "r") as f: messagebox.showinfo("Logs", f.read())

if __name__ == "__main__":
    app = App()
    app.mainloop()