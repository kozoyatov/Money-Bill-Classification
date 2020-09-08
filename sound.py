# Import the required module for text
# to speech conversion
from gtts import gTTS
from playsound import playsound
import os

pycharm_project_path='C:/Users/Daniel/Desktop/final_proj_git/final_project/svm_single_bill_classifier_0032_23_03_2020/'

swap_language_dict={
        # Chinese
        'zh-cn': 'Chinese (Mandarin/China)',
        'zh-tw': 'Chinese (Mandarin/Taiwan)',
        # English
        'en-us': 'English (US)',
        'en-ca': 'English (Canada)',
        'en-uk': 'English (UK)',
        'en-gb': 'English (UK)',
        'en-au': 'English (Australia)',
        'en-gh': 'English (Ghana)',
        'en-in': 'English (India)',
        'en-ie': 'English (Ireland)',
        'en-nz': 'English (New Zealand)',
        'en-ng': 'English (Nigeria)',
        'en-ph': 'English (Philippines)',
        'en-za': 'English (South Africa)',
        'en-tz': 'English (Tanzania)',
        # French
        'fr-ca': 'French (Canada)',
        'fr-fr': 'French (France)',
        # Portuguese
        'pt-br': 'Portuguese (Brazil)',
        'pt-pt': 'Portuguese (Portugal)',
        # Spanish
        'es-es': 'Spanish (Spain)',
        'es-us': 'Spanish (United States)'
    }
heb_dict = {'20': "Esrim", '50': "hamisheem", '100': "meah", '200': 'mataim'}
language_dict = dict([(value, key) for key, value in swap_language_dict.items()])


def PlayAmount(text,language="English (US)"):
    if language is 'he':
        text=heb_dict[text]
        language=language_dict["English (US)"]
    else:
        language = language_dict[language]
    mytext = text
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save(pycharm_project_path+"/welcome_new.mp3")
    playsound(pycharm_project_path+"/welcome_new.mp3")
    os.remove(pycharm_project_path+"/welcome_new.mp3" )





# def main():
#     PlayAmount("200")
#
# if __name__=="__main__":
#     main()


# # This module is imported so that we can
# # play the converted audio
# import os
#
# # The text that you want to convert to audio
# mytext = '200'
#
# # Language in which you want to convert
# language = 'en'
#
# # Passing the text and language to the engine,
# # here we have marked slow=False. Which tells
# # the module that the converted audio should
# # have a high speed
# myobj = gTTS(text=mytext, lang=language, slow=False)
#
# # Saving the converted audio in a mp3 file named
# # welcome
# myobj.save("welcome.mp3")
#
# # Playing the converted file
# # os.system("welcome.mp3")
#
#  #playing sound from pycharm
# playsound("welcome.mp3")
#
