#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append("WavTokenizer")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
import librosa

import torchaudio
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from speechtokenizer import SpeechTokenizer
from WavTokenizer.decoder.pretrained import WavTokenizer
from audiotools import AudioSignal


def resample(audio_data: torch.Tensor, sample_rate: int):
    print("Inout sample rate:", sample_rate)
    if sample_rate == 24000:
      audio_data24k = audio_data
      audio_data16k = torch.tensor(
          librosa.resample(
              audio_data.cpu().detach().numpy(), orig_sr=sample_rate, target_sr=16000
          )
      )
    elif sample_rate == 16000:
      audio_data16k = audio_data
      audio_data24k = torch.tensor(
          librosa.resample(
              audio_data.cpu().detach().numpy(), orig_sr=sample_rate, target_sr=24000
          )
      )
    else:
      print("Resampling everything")
      audio_data16k = torch.tensor(
          librosa.resample(
              audio_data.cpu().detach().numpy(), orig_sr=sample_rate, target_sr=16000
          )
      )
      audio_data24k = torch.tensor(
          librosa.resample(
              audio_data.cpu().detach().numpy(), orig_sr=sample_rate, target_sr=24000
          )
      )

    return (audio_data16k.view(1, -1).float().to(device), 
           audio_data24k.view(1, -1).float().to(device))


def decode_tts(tokens, quantizer, n_codebooks, n_original_tokens, start_audio_token_id, end_audio_token_id):
    # find start and end indices of audio tokens
    start = torch.nonzero(tokens == start_audio_token_id)
    end = torch.nonzero(tokens == end_audio_token_id)

    start = start[0, -1] + 1 if len(start) else 0
    end = end[0, -1] if len(end) else tokens.shape[-1]

    # subtract length of original vocabulary -> tokens in range [0, 1024)
    audio_tokens = tokens[start:end] % n_original_tokens
    reminder = audio_tokens.shape[-1] % n_codebooks

    if reminder:
        # pad if last frame is incomplete
        pad_tokens = torch.zeros(n_codebooks - reminder, device="cuda")
        audio_tokens = torch.cat([audio_tokens, pad_tokens], dim=0)

    transposed = audio_tokens.view(-1, n_codebooks).t()
    codes = transposed.view(n_codebooks, 1, -1).to(device)

    audio = quantizer.decode(codes).squeeze(0)

    del tokens
    del audio_tokens
    torch.cuda.empty_cache()

    return AudioSignal(audio.detach().cpu().numpy(), quantizer.sample_rate)


def infer_text_to_audio(text, model, tokenizer, quantizer, max_seq_length=1024, top_k=20):
    text_tokenized = tokenizer(text, return_tensors="pt")
    text_input_tokens = text_tokenized["input_ids"].to(device)

    soa = tokenizer(start_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(device)
    eoa = tokenizer(end_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(device)

    text_tokens = torch.cat([text_input_tokens, soa], dim=1)
    attention_mask = torch.ones(text_tokens.size(), device=device)

    output_audio_tokens = model.generate(
        text_tokens,
        attention_mask=attention_mask,
        max_new_tokens=max_seq_length,
        top_k=top_k,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
    )

    audio_signal = decode_tts(output_audio_tokens[0], quantizer, 3, len(tokenizer), soa, eoa)

    return audio_signal


def infer_audio_to_text(audio_path, model, tokenizer, quantizer_speech, quantizer_wav, max_seq_length=1024, top_k=20):
    audio_data, sample_rate = torchaudio.load(audio_path)

    audio_16k, audio_24k = resample(audio_data, sample_rate)
    bandwidth_id = torch.tensor([0])

    codes_semantics = quantizer_speech.encode(audio_16k.reshape(1, 1, -1))
    raw_semantic_tokens = codes_semantics + len(tokenizer)
    raw_semantic_tokens = raw_semantic_tokens[:1].view(1, -1)

    _, codes = quantizer_wav.encode_infer(audio_24k, bandwidth_id=bandwidth_id)
    raw_acoustic_tokens = codes + len(tokenizer) + 1024
    raw_acoustic_tokens = raw_acoustic_tokens.view(1, -1)

    audio_tokens = torch.cat([raw_semantic_tokens, raw_acoustic_tokens], dim=1)

    soa = tokenizer(start_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(device)
    eoa = tokenizer(end_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(device)
    audio_tokens = torch.cat([soa, audio_tokens, eoa], dim=1)
    
    # text_tokens = tokenizer("is said with", return_tensors="pt")["input_ids"].to(device)
    tokens = torch.cat([audio_tokens], dim=1)

    attention_mask = torch.ones(tokens.size(), device=device)

    output_text_tokens = model.generate(
        tokens,
        attention_mask=attention_mask,
        max_new_tokens=max_seq_length,
        do_sample=True,
        temperature=0.1,
        # top_p=0.9,
        top_k=top_k,
    )

    output_text_tokens = output_text_tokens.cpu()[0]
    output_text_tokens = output_text_tokens[output_text_tokens < tokenizer(start_audio_token)["input_ids"][-1]]
    decoded_text = tokenizer.decode(output_text_tokens, skip_special_tokens=True)

    return decoded_text



# In[2]:


device = "cuda"

n_codebooks_tts = 3
n_codebooks_asr = 1

start_audio_token = "<|start_of_audio|>"
end_audio_token = "<|end_of_audio|>"
end_sequence_token = "<|end_of_text|>"

base_model = 'Vikhrmodels/Llama-3.2-3B-asr-tts-checkpoint-12000'


quantizer_speech = SpeechTokenizer.load_from_checkpoint("audiotokenizer/speechtokenizer_hubert_avg_config.json",
                                                        "audiotokenizer/SpeechTokenizer.pt")
quantizer_speech = quantizer_speech.eval().to(device)
codebook_size = quantizer_speech.quantizer.bins

quantizer_wav = WavTokenizer.from_pretrained0802("WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                                                 "audiotokenizer/WavTokenizer_small_600_24k_4096.ckpt")
quantizer_wav = quantizer_wav.to(device)


tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=".")




model = AutoModelForCausalLM.from_pretrained(
    base_model,
    cache_dir=".",
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map={"": 0}
)



# In[3]:


text = "Say 'COUNT NUMBERS FROM ONE TO TEN' with a female speaker delivers a very monotone and high-pitched speech with a fast speed in a setting with almost no noise, creating a clear and quiet recording."
audio_signal = infer_text_to_audio(text, model, tokenizer, quantizer_speech, top_k=50)
audio_signal.write("female_audio.wav")

# Инференс аудио в текст
audio_path = "audio.wav"
generated_text = infer_audio_to_text(audio_path, model, tokenizer, quantizer_speech, quantizer_wav, top_k=50)
print(generated_text)


# In[ ]:


from IPython.display import Audio
Audio(audio_path, autoplay=True)


# In[ ]:


import gradio as gr
import torch

# Подключение функций
def infer_text_to_audio(text, prompt, top_k=20, top_p=0.8, temperature=1):
    # Форматирование текста с учетом шаблона и инструкций
    max_seq_length=1024
    formatted_text = f"Say '{text.upper()}' {prompt}"
    
    # Токенизация текста
    text_tokenized = tokenizer(formatted_text, return_tensors="pt")
    text_input_tokens = text_tokenized["input_ids"].to(device)
    soa = tokenizer(start_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(device)
    eoa = tokenizer(end_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(device)
    text_tokens = torch.cat([text_input_tokens, soa], dim=1)
    attention_mask = torch.ones(text_tokens.size(), device=device)

    output_audio_tokens = model.generate(
        text_tokens,
        attention_mask=attention_mask,
        max_new_tokens=max_seq_length,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        temperature=temperature,
        no_repeat_ngram_size=3,
        #length_penalty=2.0,
        repetition_penalty=1.5,
    )
    print(output_audio_tokens[0])
    audio_signal = decode_tts(output_audio_tokens[0], quantizer_wav, 3, len(tokenizer), soa, eoa)
    output_file = "output_audio.wav"
    audio_signal.write(output_file)
    return output_file

def infer_audio_to_text(audio_path, max_seq_length=1024, top_k=200):
    audio_data, sample_rate = torchaudio.load(audio_path)
    audio = audio_data.view(1, -1).float().to(device)
    bandwidth_id = torch.tensor([0])
    codes_semantics = quantizer_speech.encode(audio.reshape(1, 1, -1))
    raw_semantic_tokens = codes_semantics + len(tokenizer)
    raw_semantic_tokens = raw_semantic_tokens[:1].view(1, -1)
    _, codes = quantizer_wav.encode_infer(audio, bandwidth_id=bandwidth_id)
    raw_acoustic_tokens = codes + len(tokenizer) + 1024
    raw_acoustic_tokens = raw_acoustic_tokens.view(1, -1)
    audio_tokens = torch.cat([raw_semantic_tokens, raw_acoustic_tokens], dim=1)
    soa = tokenizer(start_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(device)
    eoa = tokenizer(end_audio_token, return_tensors="pt")["input_ids"][:, -1:].to(device)
    audio_tokens = torch.cat([soa, audio_tokens, eoa], dim=1)
    tokens = torch.cat([audio_tokens], dim=1)
    attention_mask = torch.ones(tokens.size(), device=device)

    output_text_tokens = model.generate(
        tokens,
        attention_mask=attention_mask,
        max_new_tokens=max_seq_length,
        do_sample=True,
        temperature=0.5,
        top_k=top_k,
    )

    output_text_tokens = output_text_tokens.cpu()[0]
    output_text_tokens = output_text_tokens[output_text_tokens < tokenizer(start_audio_token)["input_ids"][-1]]
    decoded_text = tokenizer.decode(output_text_tokens, skip_special_tokens=True)
    return decoded_text
    


# Интерфейс Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Audio and Audio-to-Text Conversion")
    
    with gr.Row():
        text_input = gr.Textbox(label="Text to Say", placeholder="Enter the text to be spoken, e.g., 'Hello everyone'")
        prompt_input = gr.Textbox(
            label="Voice Instructions", 
            placeholder=
                ("with a female voice: lively, expressive, with a playful and energetic tone. The voice should be dynamic and slightly high-pitched, conveying excitement and charm. Ensure the recording is clear and crisp, with minimal background noise.")
            
        )
        audio_output = gr.Audio(label="Generated Audio", type="filepath")
    
    with gr.Row():
        # Крутилки для управления параметрами
        top_k_slider = gr.Slider(1, 200, value=20, step=1, label="Top-k")
        top_p_slider = gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="Top-p")
        temperature_slider = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Temperature")
    
    with gr.Row():
        gr.Markdown("### Generate Audio from Text and Instructions")
        text_to_audio_button = gr.Button("Generate Audio")
        text_to_audio_button.click(
            fn=infer_text_to_audio, 
            inputs=[text_input, prompt_input, top_k_slider, top_p_slider, temperature_slider], 
            outputs=audio_output
        )
    
    with gr.Row():
        audio_input = gr.Audio(label="Input Audio for Text Generation", type="filepath")
        text_output = gr.Textbox(label="Generated Text from Audio")
    
    with gr.Row():
        gr.Markdown("### Generate Text from Audio")
        audio_to_text_button = gr.Button("Generate Text")
        audio_to_text_button.click(
            fn=infer_audio_to_text, 
            inputs=[audio_input], 
            outputs=text_output
        )

demo.launch(share=True)


# In[ ]:




