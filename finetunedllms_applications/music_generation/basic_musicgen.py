import torch 
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

def generate_music(prompt, duration=10):
  """
  Generate music from a text prompt
  Args:
    prompt (str): Text description of desired music 
    duration (int): Duration in seconds
  Returns:
  Path to generated audio file
  """

  processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
  model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

  inputs = processor(
      text = [prompt],
      padding = True,
      return_tensors = "pt",
  )

  audio_values = model.generate(
      **inputs,
      max_new_tokens = duration*50,
      do_sample = True,
      guidance_scale = 3,
  )

  sampling_rate = model.config.audio_encoder.sampling_rate
  audio_numpy = audio_values[0, 0].numpy()
  output_path = "electronic_beats.wav"
  scipy.io.wavfile.write(output_path, rate = sampling_rate, data = audio_numpy)

  return output_path


if __name__ == "__main__":
  prompt = "An energetic electronic beat with synth melodies"
  output_file = generate_music(prompt, duration=30)
  print(f"Music generated and saved to {output_file}")


def combine_prompts(musical_elements):
  """
  Combine multiple musical elements into a single prompt
  """

  return " with ".join(musical_elements)

def batch_generated(prompts, durations):
  """
  Generate multiple pieces of music in batch
  """

  results = []
  for prompt, duration in zip(prompts, durations):
    output = generate_music(prompt, duration)
    results.append(output)
  return results