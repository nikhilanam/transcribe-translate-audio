from pydub import AudioSegment
import os
import whisper
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

current_directory = os.getcwd()
global input_path
global output_path

input_path = os.path.join(current_directory,  'input')
output_path = os.path.join(current_directory, 'output')

def clip_audio(num, file_name):
    full_clip = AudioSegment.from_mp3(f'{input_path}/{file_name}.mp3')

    minutes = int(num * 60 * 1000)

    clip = full_clip[:minutes]

    clip.export(f'{output_path}/clipped/{file_name}_{int(num*60)}.mp3', format='mp3')


if __name__=='__main__':
    file_name = "german_audio"
    #clip_audio(0.5, file_name)    
    model = whisper.load_model("large")
    #audio = whisper.load_audio(f"{output_path}/clipped/{file_name}_{30}.mp3")
    audio = whisper.load_audio(f"{output_path}/clipped/german_audio_{2}.mp3")

    options = {
    "task": "transcribe" # or "transcribe" if you just want transcription
    }
    result = whisper.transcribe(model, audio, **options)
    print(result["text"])

    # translate_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    # tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    # tokenizer.src_lang = "de"

    # model_inputs = tokenizer(sample, return_tensors="pt")

    # # translate to English
    # gen_tokens = translate_model.generate(**model_inputs, forced_bos_token_id= tokenizer.get_lang_id("en"))
    # print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
    

    