import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

def generate_music(prompt, artist_style=None, duration=10, model_size="large"):
    """
    Generate music with specific artist style
    Args:
        prompt (str): Text description of desired music
        artist_style (str): Artist to emulate (e.g., "Diljit Dosanjh", "Taylor Swift")
        duration (int): Duration in seconds
        model_size (str): Size of model to use ("small", "medium", "large")
    """
    model_name = f"facebook/musicgen-{model_size}"
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    if artist_style:
        full_prompt = f"Generate music in the style of {artist_style}: {prompt}"
        if artist_style.lower() == "diljit dosanjh":
            full_prompt += " with Punjabi beats, dhol drums, and modern pop production"
        elif artist_style.lower() == "taylor swift":
            full_prompt += " with pop country fusion, strong vocals, and narrative lyrics"
    else:
        full_prompt = prompt
    
    inputs = processor(
        text=[full_prompt],
        padding=True,
        return_tensors="pt",
    )
    
    audio_values = model.generate(
        **inputs,
        max_new_tokens=duration*50,
        do_sample=True,
        guidance_scale=3.5,
        temperature=0.8
    )
    
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_numpy = audio_values[0, 0].numpy()
    
    artist_tag = f"_{artist_style.replace(' ', '_').lower()}" if artist_style else ""
    output_path = f"generated_music{artist_tag}.wav"
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_numpy)
    
    return output_path


def generate_punjabi_song(prompt, duration=30):
    return generate_music(prompt, artist_style="Diljit Dosanjh", duration=duration, model_size="large")

def generate_swift_song(prompt, duration=30):
    return generate_music(prompt, artist_style="Taylor Swift", duration=duration, model_size="large")

if __name__ == "__main__":
    punjabi_prompt = "A upbeat bhangra song with dhol drums and modern pop elements"
    punjabi_file = generate_punjabi_song(punjabi_prompt)
    print(f"Punjabi song generated: {punjabi_file}")
    
    swift_prompt = "A heartbreak pop song with country influences"
    swift_file = generate_swift_song(swift_prompt)
    print(f"Taylor Swift style song generated: {swift_file}")