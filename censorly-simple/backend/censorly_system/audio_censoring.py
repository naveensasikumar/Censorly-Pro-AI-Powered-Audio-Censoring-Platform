import numpy as np
import pandas as pd
import pickle
import re
from typing import Dict, List, Tuple
from pydub.generators import Sine
from pydub import AudioSegment
import os
from censorly_system.text_detector import AdvancedOffensiveTextDetector

class AudioCensoringSystem:
    def __init__(self, censorly_detector_path: str = "advanced_offensive_detector.pkl"):
        """Initialize audio censoring system with Advanced Censorly"""
        self.censorly_detector = AdvancedOffensiveTextDetector(censorly_detector_path)
        self.censorly_detector.load_model()
        self.whisper_model = None
        self.load_whisper_model()
        
    def load_whisper_model(self, model_size: str = "base"):
        """Load Whisper model for speech recognition"""
        print(f"Loading Whisper model ({model_size})...")
        import whisper
        self.whisper_model = whisper.load_model(model_size)
        print("Whisper model loaded successfully")
    
    def transcribe_audio_with_timestamps(self, audio_path: str) -> Dict:
        """Transcribe audio and get word-level timestamps"""
        print("Transcribing audio with word-level timestamps...")
        result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
        return result
    
    def analyze_transcription_with_censorly(self, transcription: Dict) -> List[Dict]:
        """Analyze transcription using Advanced Censorly to find offensive words"""
        offensive_segments = []
        
        for segment in transcription['segments']:
            if 'words' not in segment:
                continue
            
            sentence = segment['text'].strip()
            # Use the advanced detector with unknown word handling
            censorly_result = self.censorly_detector.analyze_text(sentence)
            
            if censorly_result['detected_words']:
                for detected_word in censorly_result['detected_words']:
                    word_to_find = detected_word['word']
                    
                    for word_info in segment['words']:
                        word_text = word_info['word'].strip().lower()
                        if word_to_find in word_text or word_text in word_to_find:
                            offensive_segments.append({
                                'word': word_info['word'],
                                'start': word_info['start'],
                                'end': word_info['end'],
                                'severity': detected_word['severity'],
                                'confidence': detected_word['confidence'],
                                'source': detected_word.get('source', 'dataset'),  # Track if live-classified
                                'sentence': sentence
                            })
                            break
        
        return offensive_segments
    
    def generate_replacement_audio(self, duration_ms: int, replacement_type: str = "beep") -> AudioSegment:
        """Generate replacement audio for censoring"""
     
        
        if replacement_type == "beep":
            frequency = 1000
            beep = Sine(frequency).to_audio_segment(duration=duration_ms)
            return beep.apply_gain(-10)
        
        elif replacement_type == "silence":
            return AudioSegment.silent(duration=duration_ms)
        
        elif replacement_type == "white_noise":
            # Generate white noise
            import numpy as np
            sample_rate = 22050
            samples = int(sample_rate * duration_ms / 1000)
            noise_data = np.random.normal(0, 0.1, samples)
            noise_data = (noise_data * 32767).astype(np.int16)
            noise = AudioSegment(
                noise_data.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            return noise.apply_gain(-20)
        
        else:
            return AudioSegment.silent(duration=duration_ms)
    
    def load_custom_replacement_audio(self, audio_file_path: str) -> AudioSegment:
        """Load custom audio file for replacement"""
        try:
            from pydub import AudioSegment
            replacement_audio = AudioSegment.from_file(audio_file_path)
            return replacement_audio
        except Exception as e:
            print(f"Error loading replacement audio: {e}")
            return None
    
    def censor_audio_segments(self, audio_path: str, offensive_segments: List[Dict], 
                            output_path: str, replacement_type: str = "beep",
                            custom_audio_path: str = None) -> Dict:
        """Censor offensive audio segments"""
        from pydub import AudioSegment
        import os
        
        print(f"Censoring audio with {len(offensive_segments)} offensive segments...")
        
        audio = AudioSegment.from_file(audio_path)
        original_duration = len(audio)
        
        if not offensive_segments:
            print("No offensive content detected")
            audio.export(output_path, format="mp3")
            return {"censored_segments": 0, "original_duration": original_duration}
        
        custom_replacement = None
        if custom_audio_path:
            custom_replacement = self.load_custom_replacement_audio(custom_audio_path)
        
        segments_sorted = sorted(offensive_segments, key=lambda x: x['start'], reverse=True)
        
        censoring_log = []
        live_classified_count = 0
        
        for segment in segments_sorted:
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            duration_ms = end_ms - start_ms
            
            if custom_replacement:
                if len(custom_replacement) >= duration_ms:
                    replacement = custom_replacement[:duration_ms]
                else:
                    replacement = custom_replacement * (duration_ms // len(custom_replacement) + 1)
                    replacement = replacement[:duration_ms]
            else:
                replacement = self.generate_replacement_audio(duration_ms, replacement_type)
            
            audio = audio[:start_ms] + replacement + audio[end_ms:]
            
            # Track if this was a live-classified word
            if segment.get('source', '').startswith('live_'):
                live_classified_count += 1
            
            censoring_log.append({
                'word': segment['word'],
                'severity': segment['severity'],
                'start_time': segment['start'],
                'end_time': segment['end'],
                'duration': duration_ms / 1000.0,
                'source': segment.get('source', 'dataset')
            })
            
            print(f"Censored '{segment['word']}' ({segment['severity']}) at {segment['start']:.2f}s-{segment['end']:.2f}s")
        
        audio.export(output_path, format="mp3")
        
        return {
            "censored_segments": len(offensive_segments),
            "live_classified_words": live_classified_count,
            "original_duration": original_duration / 1000.0,
            "output_file": output_path,
            "censoring_log": censoring_log
        }
    
    def process_audio_file(self, input_audio_path: str, output_audio_path: str = None,
                          replacement_type: str = "beep", custom_audio_path: str = None,
                          severity_filter: str = "MEDIUM") -> Dict:
        """Complete audio processing pipeline with Advanced Censorly"""
        import os
        
        if output_audio_path is None:
            name, ext = os.path.splitext(input_audio_path)
            output_audio_path = f"{name}_censored{ext}"
        
        print(f"Processing audio file: {input_audio_path}")
        
        # Transcribe with timestamps
        transcription = self.transcribe_audio_with_timestamps(input_audio_path)
        
        # Analyze with Advanced Censorly (handles unknown words automatically)
        print("Analyzing transcription with Advanced Censorly...")
        offensive_segments = self.analyze_transcription_with_censorly(transcription)
        
        # Filter by severity
        filtered_segments = []
        if severity_filter == "CRITICAL":
            filtered_segments = [s for s in offensive_segments if s['severity'] == 'CRITICAL']
        elif severity_filter == "MEDIUM":
            filtered_segments = [s for s in offensive_segments if s['severity'] in ['CRITICAL', 'MEDIUM']]
        elif severity_filter == "LOW":
            filtered_segments = [s for s in offensive_segments if s['severity'] in ['CRITICAL', 'MEDIUM', 'LOW']]
        else:  # Include all
            filtered_segments = offensive_segments
        
        # Censor the audio
        censoring_result = self.censor_audio_segments(
            input_audio_path, filtered_segments, output_audio_path, 
            replacement_type, custom_audio_path
        )
        
        # Get unknown word statistics
        unknown_stats = self.censorly_detector.get_unknown_word_stats()
        
        result = {
            "input_file": input_audio_path,
            "output_file": output_audio_path,
            "transcription": transcription['text'],
            "total_offensive_found": len(offensive_segments),
            "censored_count": len(filtered_segments),
            "severity_filter": severity_filter,
            "replacement_type": replacement_type,
            "censoring_details": censoring_result,
            "unknown_words_learned": unknown_stats.get('total_unknown_words', 0),
            "live_classified_in_this_audio": censoring_result.get('live_classified_words', 0)
        }
        
        return result
    
    def batch_process_audio_files(self, audio_files: List[str], 
                              severity_filter: str = "MEDIUM", 
                              replacement_type: str = "beep",
                              custom_audio_path: str = None,
                              output_dir: str = None,
                              **kwargs) -> List[Dict]:
        """Process multiple audio files with configurable settings"""


        print(f"Batch processing {len(audio_files)} audio files...")
        print(f"Settings: Severity={severity_filter}, Replacement={replacement_type}")
        if output_dir:
            print(f"Output directory: {output_dir}")
        print("=" * 60)

        results = []

        for i, audio_file in enumerate(audio_files):
            print(f"\nProcessing file {i+1}/{len(audio_files)}: {audio_file}")

            try:
                # Generate output path
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.basename(audio_file)
                    name, ext = os.path.splitext(filename)
                    output_audio_path = os.path.join(output_dir, f"{name}_censored{ext}")
                else:
                    output_audio_path = kwargs.get('output_audio_path', None)

                # Process the file
                result = self.process_audio_file(
                    input_audio_path=audio_file,
                    output_audio_path=output_audio_path,
                    replacement_type=replacement_type,
                    custom_audio_path=custom_audio_path,
                    severity_filter=severity_filter,
                    **kwargs
                )

                results.append(result)

                print(f"  Success: {result['censored_count']} words censored")
                if result['unknown_words_learned'] > 0:
                    print(f"  Learned {result['unknown_words_learned']} unknown words")

            except Exception as e:
                print(f"  Error processing {audio_file}: {e}")
                results.append({
                    "error": str(e),
                    "file": audio_file,
                    "settings": {
                        "severity_filter": severity_filter,
                        "replacement_type": replacement_type
                    }
                })

        # Batch summary
        print(f"\n{'=' * 60}")
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)

        successful_files = [r for r in results if 'error' not in r]
        failed_files = [r for r in results if 'error' in r]

        print(f"Total files processed: {len(audio_files)}")
        print(f"Successful: {len(successful_files)}")
        print(f"Failed: {len(failed_files)}")
        print(f"Success rate: {len(successful_files) / len(audio_files) * 100:.1f}%")

        if successful_files:
            total_censored = sum(r['censored_count'] for r in successful_files)
            total_unknown = sum(r['unknown_words_learned'] for r in successful_files)

            print(f"Total words censored: {total_censored}")
            print(f"Total unknown words learned: {total_unknown}")

            severity_counts = {}
            for result in successful_files:
                for log_entry in result['censoring_details']['censoring_log']:
                    severity = log_entry['severity']
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

            if severity_counts:
                print("Words censored by severity:")
                for severity in ['CRITICAL', 'MEDIUM', 'LOW']:
                    if severity in severity_counts:
                        print(f"  {severity}: {severity_counts[severity]}")

        if failed_files:
            print("\nFailed files:")
            for failed in failed_files:
                print(f"  - {failed['file']}: {failed['error']}")

        return results
