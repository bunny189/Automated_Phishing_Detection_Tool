# utils/feature_extraction.py
import re
import tldextract
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import math

SUSPICIOUS_TOKENS = [
    "login", "signin", "bank", "secure", "account", "update", "confirm", "verify",
    "ebayisapi", "webscr", "paypal", "appleid"
]

def normalize_url(u: str) -> str:
    u = u.strip()
    if not re.match(r"^https?://", u):
        u = "http://" + u
    return u

def domain_and_subdomain(url):
    ext = tldextract.extract(url)
    domain = ext.domain + '.' + ext.suffix if ext.suffix else ext.domain
    subdomain = ext.subdomain
    return domain or "", subdomain or ""

def count_digits(s):
    return sum(c.isdigit() for c in s)

def count_special_chars(s):
    return sum(1 for c in s if not c.isalnum())

def get_basic_url_features(urls):
    """Return numeric feature matrix (list of feature lists) for given URL list."""
    feats = []
    for u in urls:
        u_norm = normalize_url(u)
        parsed = urlparse(u_norm)
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        query = parsed.query or ""
        full = netloc + path + query

        domain, subdomain = domain_and_subdomain(u_norm)
        domain_parts = domain.split('.') if domain else []
        subdomain_parts = subdomain.split('.') if subdomain else []

        f_len = len(u_norm)
        f_len_path = len(path)
        f_has_https = 1 if parsed.scheme == "https" else 0
        f_count_dots = netloc.count('.')
        f_count_hyphens = netloc.count('-') + path.count('-')
        f_count_at = 1 if '@' in u_norm else 0
        f_digits = count_digits(u_norm)
        f_special = count_special_chars(u_norm)
        f_subdomain_parts = len([p for p in subdomain_parts if p])
        f_domain_parts = len([p for p in domain_parts if p])
        f_suspicious_tokens = sum(1 for t in SUSPICIOUS_TOKENS if t in u_norm.lower())
        # entropy-like measure for domain
        char_set = set(netloc)
        f_entropy = 0.0
        if len(netloc) > 0:
            probs = [netloc.count(ch)/len(netloc) for ch in char_set]
            f_entropy = -sum(p*math.log(p+1e-9) for p in probs)

        feats.append([
            f_len, f_len_path, f_has_https, f_count_dots, f_count_hyphens,
            f_count_at, f_digits, f_special, f_subdomain_parts, f_domain_parts,
            f_suspicious_tokens, f_entropy
        ])
    return feats

def fetch_page_text(url, timeout=3):
    """Optional: fetch page and return visible text. Used sparingly (may be unreliable)."""
    try:
        r = requests.get(normalize_url(url), timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.content, "lxml")
        # remove scripts/styles
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:2000]  # cap length
    except Exception:
        return ""

def url_tokenize_for_vector(url):
    """Return a text-like token string from URL to feed into TF-IDF."""
    u = normalize_url(url).lower()
    # break by non-alphanumeric and dots/slashes/hyphens
    tokens = re.split(r"[^a-z0-9]", u)
    tokens = [t for t in tokens if t and len(t) > 1]
    return " ".join(tokens)
