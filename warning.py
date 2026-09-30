import pyttsx3

def give_voice_warning(warning_num):
    engine = pyttsx3.init()

    text = f"Please remove your mask. Warning {warning_num}"
    engine.say(text)
    engine.runAndWait()
