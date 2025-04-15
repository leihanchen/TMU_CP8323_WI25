from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_slide_content(file_path):
    """
    Process a text file and extract content after each numerical line.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of strings, each containing content after a numerical line
    """
    slides_content = []
    current_content = []
    is_collecting = False
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Check if the line contains only a number (possibly with whitespace)
            if line.strip().isdigit():
                # If we were already collecting content, save it
                if is_collecting and current_content:
                    slides_content.append('\n'.join(current_content).strip())
                    current_content = []
                
                # Start collecting for the next slide
                is_collecting = True
            elif is_collecting:
                # If not a number and we're collecting, add to current content
                current_content.append(line.strip())
    
    # Add the last slide if there's content
    if current_content:
        slides_content.append('\n'.join(current_content).strip())
    
    return slides_content

def generate_audio_for_slides(scripts, client, voice_id="pNInz6obpgDQGcFmaJgB", speed=0.0, output_folder="slide_audio"):
    """
    Generate audio files from scripts with adjusted speaking rate.
    
    Args:
        scripts: List of text scripts for each slide
        client: ElevenLabs client instance
        voice_id: Voice ID to use (default is "pNInz6obpgDQGcFmaJgB")
        speed: Voice speed (0.7 to 1.2)
             0.7: Very slow
             0.0: Normal speed (default)
             1.2: Very fast
        output_folder: Folder to save audio files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Configure voice settings
    voice_settings = {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
        "speaking_rate": speed  # This controls the speed
    }
    
    # Generate audio file for each slide script
    for i, script in enumerate(scripts):
        audio = client.text_to_speech.convert(
            text=script, 
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            voice_settings=voice_settings
        )
        
        output_file = os.path.join(output_folder, f"slide_{i+1}.mp3")
        # Writing the audio to a file
        with open(output_file, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        print(f"Generated audio for slide {i+1} with speed {speed}")

if __name__ == "__main__":
    # Extract slides content from the text file
    slides_file_path = "slides_notes.txt"
    slide_scripts = extract_slide_content(slides_file_path)
    
    # Print number of slides found
    print(f"Found {len(slide_scripts)} slides in the file")
    
    # Initialize ElevenLabs client
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not found")
        exit(1)
    
    client = ElevenLabs(api_key=api_key)
    
    # Generate audio for each slide
    # You can adjust the speed parameter as needed (0.7 to 1.2)
    generate_audio_for_slides(slide_scripts, client, speed=1.0)
    
    print("All slide audio files have been generated!")