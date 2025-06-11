import numpy as np
import pandas as pd
import pickle
import re
from typing import Dict, List, Tuple

class AdvancedOffensiveTextDetector:
    def __init__(self, model_path: str = "advanced_offensive_detector.pkl"):
        """Advanced text detector with natural usage-based classifications"""
        self.model_path = model_path
        self.word_dataset = None
        self.word_lookup = None
        self.usage_dataset = None
        self.live_classifier = None  # For unknown words
        
        self.severity_hierarchy = {
            'CRITICAL': 3,
            'MEDIUM': 2, 
            'LOW': 1,
            'UNOFFENSIVE': 0
        }
        
        # Cache for dynamically classified words
        self.dynamic_cache = {}
    
    def load_word_dataset(self, word_df: pd.DataFrame, usage_df: pd.DataFrame = None):
        """Load the natural usage-classified word dataset"""
        self.word_dataset = word_df
        self.usage_dataset = usage_df
        
        self.word_lookup = {}
        for _, row in word_df.iterrows():
            word = row['word'].lower()
            self.word_lookup[word] = {
                'severity': row['severity'],
                'confidence': row['confidence'],
                'natural_usage_score': row.get('natural_usage_score', 0.5),
                'avg_toxicity': row.get('avg_toxicity', 0.0),
                'sentences_analyzed': row.get('sentences_analyzed', 0),
                'high_toxicity_percentage': row.get('high_toxicity_percentage', 0.0)
            }
        
        # Initialize live classifier for unknown words
        try:
            from transformers import pipeline
            self.live_classifier = AdvancedMLWordClassifier("live_classifier_cache.pkl")
            print(f"Live classifier loaded for unknown words")
        except Exception as e:
            print(f"Live classifier not available: {e}")
        
        print(f"Loaded {len(self.word_lookup)} natural usage-classified word variations")
        if usage_df is not None:
            print(f"Loaded {len(usage_df)} natural usage examples")
        
        severity_counts = word_df['severity'].value_counts()
        for severity in ['CRITICAL', 'MEDIUM', 'LOW', 'UNOFFENSIVE']:
            count = severity_counts.get(severity, 0)
            print(f"  {severity}: {count}")
    
    def classify_unknown_word(self, word: str) -> Dict[str, any]:
        """Classify a word not in the dataset using live ML classification"""
        
        # Check cache first
        if word in self.dynamic_cache:
            print(f"Using cached classification for '{word}'")
            return self.dynamic_cache[word]
        
        print(f"🔍 Unknown word detected: '{word}' - running live classification...")
        
        if self.live_classifier is None:
            # Fallback to basic heuristics
            return self._fallback_classification(word)
        
        try:
            # Use the live classifier for unknown words
            prediction, confidence, analysis = self.live_classifier.classify_word_with_natural_usage(word)
            
            word_info = {
                'severity': prediction,
                'confidence': confidence,
                'usage_score': analysis.get('natural_usage_score', 0.5),
                'avg_toxicity': analysis.get('avg_toxicity', 0.0),
                'sentences_analyzed': analysis.get('total_sentences_analyzed', 0),
                'classification_method': 'live_ml'
            }
            
            # Cache the result
            self.dynamic_cache[word] = word_info
            
            print(f"Live classification: '{word}' → {prediction} (confidence: {confidence:.3f})")
            return word_info
            
        except Exception as e:
            print(f"Live classification failed for '{word}': {e}")
            return self._fallback_classification(word)
    
    def _fallback_classification(self, word: str) -> Dict[str, any]:
        """Fallback classification using heuristics when ML is unavailable"""
        
        # Known severe profanity patterns (be conservative)
        severe_patterns = ['f*ck', 'sh*t', 'b*tch', 'n*gger', 'c*nt']
        moderate_patterns = ['damn', 'hell', 'crap', 'ass']
        mild_patterns = ['stupid', 'dumb', 'idiot', 'fool', 'crazy']
        
        word_lower = word.lower()
        
        # Check for severe patterns
        for pattern in severe_patterns:
            clean_pattern = pattern.replace('*', '')
            if clean_pattern in word_lower or word_lower in clean_pattern:
                severity = 'CRITICAL'
                confidence = 0.8
                break
        else:
            # Check for moderate patterns
            for pattern in moderate_patterns:
                if pattern in word_lower or word_lower in pattern:
                    severity = 'MEDIUM'
                    confidence = 0.6
                    break
            else:
                # Check for mild patterns
                for pattern in mild_patterns:
                    if pattern in word_lower or word_lower in pattern:
                        severity = 'LOW'
                        confidence = 0.5
                        break
                else:
                    # Default to LOW for unknown words (conservative approach)
                    severity = 'LOW'
                    confidence = 0.3
        
        word_info = {
            'severity': severity,
            'confidence': confidence,
            'usage_score': 0.4,
            'avg_toxicity': 0.4,
            'sentences_analyzed': 0,
            'classification_method': 'heuristic_fallback'
        }
        
        # Cache the result
        self.dynamic_cache[word] = word_info
        
        print(f"🔧 Heuristic classification: '{word}' → {severity} (confidence: {confidence:.3f})")
        return word_info

    def analyze_text(self, text: str) -> Dict[str, any]:
        """Analyze text using natural usage-enhanced classifications with unknown word handling"""
        if self.word_lookup is None:
            raise ValueError("Word dataset not loaded. Call load_word_dataset() first.")
        
        detected_words = []
        unknown_words = []
        text_lower = text.lower()
        
        # Find exact word matches
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if word in self.word_lookup:
                # Known word from dataset
                detected_words.append({
                    'word': word,
                    'severity': self.word_lookup[word]['severity'],
                    'confidence': self.word_lookup[word]['confidence'],
                    'usage_score': self.word_lookup[word]['natural_usage_score'],
                    'avg_toxicity': self.word_lookup[word]['avg_toxicity'],
                    'match_type': 'exact',
                    'source': 'dataset'
                })
            else:
                # Check if it might be profanity (simple heuristics)
                if self._might_be_profanity(word):
                    unknown_words.append(word)
        
        # Classify unknown words that might be profanity
        for word in unknown_words:
            word_info = self.classify_unknown_word(word)
            detected_words.append({
                'word': word,
                'severity': word_info['severity'],
                'confidence': word_info['confidence'],
                'usage_score': word_info['usage_score'],
                'avg_toxicity': word_info['avg_toxicity'],
                'match_type': 'exact',
                'source': f"live_{word_info['classification_method']}"
            })
        
        # Find partial matches for variations (symbols, numbers, spaces)
        for variation, info in self.word_lookup.items():
            if variation in text_lower and variation not in words:
                if (len(variation) > 3 and 
                    ('*' in variation or '@' in variation or '$' in variation or 
                     any(char.isdigit() for char in variation) or ' ' in variation)):
                    
                    pattern = r'\b' + re.escape(variation) + r'\b'
                    if re.search(pattern, text_lower):
                        detected_words.append({
                            'word': variation,
                            'severity': info['severity'],
                            'confidence': info['confidence'] * 0.8,  # Slightly lower for partial matches
                            'usage_score': info['natural_usage_score'],
                            'avg_toxicity': info['avg_toxicity'],
                            'match_type': 'partial',
                            'source': 'dataset'
                        })
        
        overall_prediction, overall_confidence = self.calculate_overall_severity(detected_words)
        
        # Categorize detected words by severity
        critical_words = [w for w in detected_words if w['severity'] == 'CRITICAL']
        medium_words = [w for w in detected_words if w['severity'] == 'MEDIUM']
        low_words = [w for w in detected_words if w['severity'] == 'LOW']
        unoffensive_words = [w for w in detected_words if w['severity'] == 'UNOFFENSIVE']
        
        # Track unknown word statistics
        live_classified = [w for w in detected_words if w['source'].startswith('live_')]
        
        return {
            'text': text,
            'prediction': overall_prediction,
            'confidence': overall_confidence,
            'detected_words': detected_words,
            'unknown_words_found': len(unknown_words),
            'live_classified_words': len(live_classified),
            'severity_breakdown': {
                'critical_count': len(critical_words),
                'medium_count': len(medium_words),
                'low_count': len(low_words),
                'unoffensive_count': len(unoffensive_words),
                'total_detected': len(detected_words)
            },
            'severity_details': {
                'critical_words': [w['word'] for w in critical_words],
                'medium_words': [w['word'] for w in medium_words],
                'low_words': [w['word'] for w in low_words],
                'unoffensive_words': [w['word'] for w in unoffensive_words]
            },
            'source_breakdown': {
                'dataset_words': len([w for w in detected_words if w['source'] == 'dataset']),
                'live_ml_words': len([w for w in detected_words if 'live_ml' in w['source']]),
                'heuristic_words': len([w for w in detected_words if 'heuristic' in w['source']])
            },
            'enhanced_features': {
                'avg_usage_score': np.mean([w['usage_score'] for w in detected_words]) if detected_words else 0.0,
                'avg_toxicity': np.mean([w['avg_toxicity'] for w in detected_words]) if detected_words else 0.0,
                'natural_usage_enhanced': True,
                'unknown_word_handling': True
            }
        }
    
    def _might_be_profanity(self, word: str) -> bool:
        """Simple heuristic to determine if an unknown word might be profanity"""
        
        # Skip very short or very long words
        if len(word) < 3 or len(word) > 15:
            return False
        
        # Skip common safe words
        safe_words = {
            'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but', 'his', 'from',
            'they', 'she', 'her', 'been', 'than', 'its', 'who', 'did', 'yes', 'get', 'may', 'him',
            'old', 'see', 'now', 'way', 'could', 'people', 'my', 'than', 'first', 'water', 'been',
            'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made',
            'very', 'good', 'nice', 'great', 'awesome', 'amazing', 'wonderful', 'excellent', 'perfect',
            'hello', 'thanks', 'please', 'sorry', 'welcome', 'goodbye'
        }
        
        if word.lower() in safe_words:
            return False
        
        # Indicators that might suggest profanity
        profanity_indicators = [
            # Contains substitution characters
            any(char in word for char in ['*', '@', '$', '!', '3', '1', '0']),
            # Repeated characters (like "shiiiit")
            any(word.count(char) >= 3 for char in set(word) if char.isalpha()),
            # Mixed case in unusual patterns
            sum(1 for c in word if c.isupper()) > 1 and not word.isupper(),
            # Contains common profanity letter patterns
            any(pattern in word.lower() for pattern in ['ck', 'sh', 'fk', 'bt', 'dmn'])
        ]
        
        # If multiple indicators are present, classify it
        return sum(profanity_indicators) >= 2
    
    def calculate_overall_severity(self, detected_words: List[Dict]) -> Tuple[str, float]:
        """Calculate severity with natural usage weighting"""
        if not detected_words:
            return 'UNOFFENSIVE', 0.9
        
        severity_scores = {'CRITICAL': [], 'MEDIUM': [], 'LOW': [], 'UNOFFENSIVE': []}
        usage_scores = {'CRITICAL': [], 'MEDIUM': [], 'LOW': [], 'UNOFFENSIVE': []}
        
        for word_info in detected_words:
            severity = word_info['severity']
            confidence = word_info['confidence']
            usage_score = word_info.get('usage_score', 0.5)
            
            if severity in severity_scores:
                severity_scores[severity].append(confidence)
                usage_scores[severity].append(usage_score)
        
        critical_count = len(severity_scores['CRITICAL'])
        medium_count = len(severity_scores['MEDIUM'])
        low_count = len(severity_scores['LOW'])
        unoffensive_count = len(severity_scores['UNOFFENSIVE'])
        
        # Critical words always result in critical classification
        if critical_count > 0:
            avg_confidence = np.mean(severity_scores['CRITICAL'])
            avg_usage = np.mean(usage_scores['CRITICAL'])
            final_confidence = (avg_confidence * 0.7) + (avg_usage * 0.3)
            return 'CRITICAL', min(0.95, final_confidence + 0.1)
        
        # Multiple medium words escalate to critical
        elif medium_count >= 2:
            avg_confidence = np.mean(severity_scores['MEDIUM'])
            avg_usage = np.mean(usage_scores['MEDIUM'])
            final_confidence = (avg_confidence * 0.7) + (avg_usage * 0.3)
            return 'CRITICAL', min(0.9, final_confidence)
        
        # Single medium word stays medium
        elif medium_count == 1:
            confidence = max(severity_scores['MEDIUM'])
            usage = max(usage_scores['MEDIUM'])
            final_confidence = (confidence * 0.8) + (usage * 0.2)
            return 'MEDIUM', final_confidence
        
        # Many low words can escalate to medium
        elif low_count >= 3:
            avg_confidence = np.mean(severity_scores['LOW'])
            avg_usage = np.mean(usage_scores['LOW'])
            final_confidence = (avg_confidence * 0.7) + (avg_usage * 0.3)
            return 'MEDIUM', min(0.8, final_confidence + 0.15)
        
        # Few low words stay low
        elif low_count > 0:
            avg_confidence = np.mean(severity_scores['LOW'])
            avg_usage = np.mean(usage_scores['LOW'])
            final_confidence = (avg_confidence * 0.8) + (avg_usage * 0.2)
            return 'LOW', final_confidence
        
        # Only unoffensive words detected
        elif unoffensive_count > 0:
            return 'UNOFFENSIVE', 0.9
        
        else:
            return 'UNOFFENSIVE', 0.9
    
    def get_detailed_analysis(self, text: str) -> Dict[str, any]:
        """Get detailed analysis including usage statistics"""
        analysis = self.analyze_text(text)
        
        # Add detailed statistics
        if analysis['detected_words']:
            detected = analysis['detected_words']
            
            analysis['detailed_stats'] = {
                'word_count': len(detected),
                'avg_confidence': np.mean([w['confidence'] for w in detected]),
                'max_confidence': max([w['confidence'] for w in detected]),
                'avg_usage_score': np.mean([w['usage_score'] for w in detected]),
                'severity_distribution': {
                    'critical_percentage': len([w for w in detected if w['severity'] == 'CRITICAL']) / len(detected) * 100,
                    'medium_percentage': len([w for w in detected if w['severity'] == 'MEDIUM']) / len(detected) * 100,
                    'low_percentage': len([w for w in detected if w['severity'] == 'LOW']) / len(detected) * 100,
                    'unoffensive_percentage': len([w for w in detected if w['severity'] == 'UNOFFENSIVE']) / len(detected) * 100
                }
            }
        
        return analysis
    
    def save_model(self):
        """Save the enhanced detector model including dynamic cache"""
        model_data = {
            'word_lookup': self.word_lookup,
            'word_dataset': self.word_dataset,
            'usage_dataset': self.usage_dataset,
            'severity_hierarchy': self.severity_hierarchy,
            'dynamic_cache': self.dynamic_cache,  # Save dynamically learned words
            'model_version': '3.0_Natural_Usage_Enhanced_with_Live_Classification'
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"AdvancedOffensiveTextDetector saved to {self.model_path}")
        if self.dynamic_cache:
            print(f"Saved {len(self.dynamic_cache)} dynamically classified words")
    
    def load_model(self) -> bool:
        """Load saved enhanced detector including dynamic cache"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.word_lookup = model_data['word_lookup']
            self.word_dataset = model_data['word_dataset']
            self.usage_dataset = model_data.get('usage_dataset')
            self.severity_hierarchy = model_data.get('severity_hierarchy', self.severity_hierarchy)
            self.dynamic_cache = model_data.get('dynamic_cache', {})  # Load cached unknown words
            
            # Initialize live classifier
            try:
                self.live_classifier = AdvancedMLWordClassifier("live_classifier_cache.pkl")
            except:
                self.live_classifier = None
            
            print(f"AdvancedOffensiveTextDetector loaded from {self.model_path}")
            print(f"Loaded {len(self.word_lookup)} word variations")
            if self.dynamic_cache:
                print(f"Loaded {len(self.dynamic_cache)} previously classified unknown words")
            return True
            
        except FileNotFoundError:
            print("No saved advanced detector found")
            return False
        except Exception as e:
            print(f"Error loading advanced detector: {e}")
            return False
    
    def add_word_to_dataset(self, word: str, severity: str, confidence: float = 0.8):
        """Manually add a word to the dataset (for admin/feedback purposes)"""
        word_lower = word.lower()
        
        word_info = {
            'severity': severity,
            'confidence': confidence,
            'natural_usage_score': 0.5,
            'avg_toxicity': 0.5,
            'sentences_analyzed': 0,
            'high_toxicity_percentage': 50.0
        }
        
        self.word_lookup[word_lower] = word_info
        
        # Also add to dynamic cache
        self.dynamic_cache[word_lower] = {
            **word_info,
            'classification_method': 'manual_addition'
        }
        
        print(f"Manually added '{word}' → {severity} to dataset")
        
        # Auto-save the updated model
        self.save_model()
    
    def get_unknown_word_stats(self) -> Dict[str, any]:
        """Get statistics about unknown words encountered"""
        if not self.dynamic_cache:
            return {"message": "No unknown words encountered yet"}
        
        cache_severities = [info['severity'] for info in self.dynamic_cache.values()]
        cache_methods = [info['classification_method'] for info in self.dynamic_cache.values()]
        
        return {
            'total_unknown_words': len(self.dynamic_cache),
            'severity_distribution': {
                'CRITICAL': cache_severities.count('CRITICAL'),
                'MEDIUM': cache_severities.count('MEDIUM'),
                'LOW': cache_severities.count('LOW'),
                'UNOFFENSIVE': cache_severities.count('UNOFFENSIVE')
            },
            'classification_methods': {
                'live_ml': cache_methods.count('live_ml'),
                'heuristic_fallback': cache_methods.count('heuristic_fallback'),
                'manual_addition': cache_methods.count('manual_addition')
            },
            'unknown_words': list(self.dynamic_cache.keys())
        }