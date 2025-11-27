import pandas as pd
import re
from urllib.parse import urlparse
try:
    import tldextract
except Exception:
    tldextract = None

def _extract_domain_parts(url):
    """Return an object-like with domain, subdomain, and suffix attributes.
    Uses tldextract if available, otherwise a simple fallback using urlparse.
    """
    class Parts:
        def __init__(self, domain, subdomain, suffix):
            self.domain = domain
            self.subdomain = subdomain
            self.suffix = suffix

    if tldextract:
        try:
            return tldextract.extract(url)
        except Exception:
            pass

    # Fallback parsing
    try:
        hostname = urlparse(url).hostname or ''
        # Remove potential port
        hostname = hostname.split(':')[0]
        parts = hostname.split('.') if hostname else []
        if len(parts) >= 2:
            domain = parts[-2]
            suffix = parts[-1]
            subdomain = '.'.join(parts[:-2]) if len(parts) > 2 else ''
        elif len(parts) == 1:
            domain = parts[0]
            suffix = ''
            subdomain = ''
        else:
            domain = ''
            subdomain = ''
            suffix = ''
        return Parts(domain=domain, subdomain=subdomain, suffix=suffix)
    except Exception:
        return Parts(domain='', subdomain='', suffix='')

class URLFeatureExtractor:
    def __init__(self):
        pass
    
    def extract_features(self, url):
        """Extract all features from a URL"""
        features = {}
        
        # Basic URL features
        features['url_length'] = len(url)
        features['hostname_length'] = len(urlparse(url).hostname or '')
        features['path_length'] = len(urlparse(url).path)
        
        # Special character counts
        features['count_dot'] = url.count('.')
        features['count_hyphen'] = url.count('-')
        features['count_underscore'] = url.count('_')
        features['count_slash'] = url.count('/')
        features['count_questionmark'] = url.count('?')
        features['count_equal'] = url.count('=')
        features['count_at'] = url.count('@')
        features['count_and'] = url.count('&')
        features['count_exclamation'] = url.count('!')
        features['count_space'] = url.count(' ')
        features['count_tilde'] = url.count('~')
        features['count_comma'] = url.count(',')
        features['count_plus'] = url.count('+')
        features['count_asterisk'] = url.count('*')
        features['count_hash'] = url.count('#')
        features['count_dollar'] = url.count('$')
        features['count_percent'] = url.count('%')
        
        # Character type counts
        features['count_digits'] = sum(c.isdigit() for c in url)
        features['count_letters'] = sum(c.isalpha() for c in url)
        features['count_other_chars'] = len(url) - features['count_digits'] - features['count_letters']
        
        # Protocol features
        features['uses_https'] = 1 if url.startswith('https') else 0
        features['uses_http'] = 1 if url.startswith('http') else 0
        
        # Domain features
        try:
            extracted = _extract_domain_parts(url)
            features['domain_length'] = len(extracted.domain)
            features['subdomain_length'] = len(extracted.subdomain)
            features['tld_length'] = len(extracted.suffix)

            # Subdomain features
            features['has_subdomain'] = 1 if extracted.subdomain else 0
            features['subdomain_count'] = extracted.subdomain.count('.') + 1 if extracted.subdomain else 0
        except Exception:
            features['domain_length'] = 0
            features['subdomain_length'] = 0
            features['tld_length'] = 0
            features['has_subdomain'] = 0
            features['subdomain_count'] = 0
        
        # Path features
        path = urlparse(url).path
        features['fd_length'] = len(path.split('/')[0]) if path else 0
        features['num_directories'] = path.count('/')
        
        # Query features
        query = urlparse(url).query
        features['has_query'] = 1 if query else 0
        features['query_length'] = len(query)
        
        # Suspicious patterns
        features['ip_in_url'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
        features['is_shortened'] = 1 if any(service in url for service in 
                                           ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly']) else 0
        
        # Suspicious keywords
        suspicious_keywords = ['login', 'signin', 'verify', 'account', 'secure', 'update', 
                              'banking', 'confirm', 'password', 'wallet', 'click']
        features['suspicious_words_count'] = sum(1 for word in suspicious_keywords if word in url.lower())
        
        # Entropy (measure of randomness)
        features['entropy'] = self.calculate_entropy(url)
        
        # vowel_ratio
        vowels = sum(1 for char in url.lower() if char in 'aeiou')
        features['vowel_ratio'] = vowels / len(url) if url else 0
        
        return features
    
    def calculate_entropy(self, text):
        """Calculate Shannon entropy of the URL"""
        if not text:
            return 0
        # Use Shannon entropy: -sum(p_x * log2(p_x)) for each character probability
        import math
        # Count frequency of each character in the text
        freq = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1

        entropy = 0.0
        length = len(text)
        for count in freq.values():
            p_x = count / length
            if p_x > 0:
                entropy -= p_x * math.log2(p_x)
        return entropy

def create_feature_dataframe(urls, labels=None):
    """Create a DataFrame with features from URLs"""
    extractor = URLFeatureExtractor()
    features_list = []
    
    for i, url in enumerate(urls):
        if i % 10000 == 0:
            print(f"Processing URL {i}/{len(urls)}")
        
        features = extractor.extract_features(url)
        if labels is not None:
            features['label'] = labels.iloc[i] if hasattr(labels, 'iloc') else labels[i]
        features_list.append(features)
    
    return pd.DataFrame(features_list)