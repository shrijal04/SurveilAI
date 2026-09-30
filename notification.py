import smtplib
from email.mime.text import MIMEText
import pyttsx3

def notification_send():
    sender = "shrijalsthapit02@gmail.com"
    app_password = "itbs frvl uyaa oonv"
    receiver = "samridhdisthapit2@gmail.com"

    # Email body
    msg = MIMEText("Check the ATM — someone may not be removing their mask.")
    msg["Subject"] = "ATM Alert"
    msg["From"] = sender
    msg["To"] = receiver

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, receiver, msg.as_string())

    # Text-to-speech after email is sent
    engine = pyttsx3.init()
    engine.say("Notification Sent")
    engine.runAndWait()