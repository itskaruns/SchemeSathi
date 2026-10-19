"""States / UTs, a city->state map, and a forgiving resolver ("Udaipur", "updaipur", "pune" -> a valid state)."""
import difflib, re

STATES = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
          "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
          "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
          "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
          "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
          "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"]

_C = {
    "Maharashtra": "pune mumbai nagpur nashik aurangabad chhatrapati sambhajinagar thane kolhapur solapur amravati "
                   "nanded sangli satara latur jalgaon akola dhule ahmednagar navi mumbai pimpri chinchwad "
                   "ratnagiri baramati",
    "Rajasthan": "udaipur jaipur jodhpur kota ajmer bikaner alwar bhilwara sikar jaisalmer pali bharatpur",
    "Gujarat": "ahmedabad surat vadodara baroda rajkot bhavnagar jamnagar gandhinagar anand junagadh",
    "Karnataka": "bengaluru bangalore mysuru mysore hubli dharwad mangaluru mangalore belagavi belgaum davanagere",
    "Tamil Nadu": "chennai coimbatore madurai trichy tiruchirappalli salem tirunelveli vellore erode thanjavur",
    "Kerala": "kochi cochin thiruvananthapuram trivandrum kozhikode calicut thrissur kannur kollam",
    "Uttar Pradesh": "lucknow kanpur varanasi agra prayagraj allahabad meerut noida ghaziabad bareilly aligarh gorakhpur "
                     "mathura",
    "West Bengal": "kolkata calcutta howrah durgapur siliguri asansol kharagpur",
    "Delhi": "new delhi delhi",
    "Madhya Pradesh": "bhopal indore gwalior jabalpur ujjain sagar rewa",
    "Bihar": "patna gaya muzaffarpur bhagalpur darbhanga",
    "Telangana": "hyderabad secunderabad warangal nizamabad karimnagar",
    "Andhra Pradesh": "visakhapatnam vizag vijayawada guntur tirupati nellore kurnool amaravati",
    "Odisha": "bhubaneswar cuttack rourkela puri sambalpur",
    "Punjab": "ludhiana amritsar jalandhar patiala mohali bathinda",
    "Haryana": "gurugram gurgaon faridabad panipat ambala rohtak hisar karnal",
    "Chandigarh": "chandigarh",
    "Assam": "guwahati dibrugarh silchar jorhat",
    "Jharkhand": "ranchi jamshedpur dhanbad bokaro",
    "Chhattisgarh": "raipur bhilai bilaspur durg",
    "Uttarakhand": "dehradun haridwar roorkee nainital haldwani",
    "Himachal Pradesh": "shimla manali dharamshala solan",
    "Goa": "panaji margao vasco",
    "Jammu and Kashmir": "srinagar jammu",
    "Puducherry": "pondicherry puducherry",
}
CITY_STATE = {}
for _st, _cities in _C.items():
    # multi-word cities are stored whole; single words individually
    for _c in re.split(r"\s+", _cities):
        CITY_STATE[_c] = _st
for _multi, _st in [("navi mumbai", "Maharashtra"), ("new delhi", "Delhi"), ("pimpri chinchwad", "Maharashtra")]:
    CITY_STATE[_multi] = _st
ALIASES = {"orissa": "Odisha", "pondicherry": "Puducherry", "j&k": "Jammu and Kashmir", "up": "Uttar Pradesh",
           "mp": "Madhya Pradesh", "tn": "Tamil Nadu", "wb": "West Bengal", "ap": "Andhra Pradesh",
           "maharastra": "Maharashtra", "mah": "Maharashtra"}
_LOW = {s.lower(): s for s in STATES}


def normalize_state(value):
    """Return a valid state name or None. Accepts a state, a city, an alias, or a near-miss typo."""
    if not value or not isinstance(value, str):
        return None
    v = re.sub(r"[^a-z& ]", " ", value.lower()).strip()
    v = re.sub(r"\s+", " ", v)
    if not v:
        return None
    for table in (_LOW, {k: v_ for k, v_ in ALIASES.items()}, CITY_STATE):
        if v in table:
            return table[v] if table is not _LOW else _LOW[v]
    pool = list(_LOW) + list(CITY_STATE) + [a for a in ALIASES if len(a) > 3]
    hit = difflib.get_close_matches(v, pool, n=1, cutoff=0.8)
    return normalize_state(hit[0]) if hit and hit[0] != v else None


def find_state_in_text(text: str):
    """Scan free text for a state or city mention (longest names first, so 'new delhi' beats 'delhi')."""
    low = " " + re.sub(r"[^a-z& ]", " ", text.lower()) + " "
    low = re.sub(r"\s+", " ", low)
    names = sorted(list(_LOW) + list(CITY_STATE), key=len, reverse=True)
    for n in names:
        if f" {n} " in low:
            return _LOW.get(n) or CITY_STATE[n]
    # typo tolerance: compare each word (and word pairs) against known names
    words = low.split()
    cand = words + [" ".join(p) for p in zip(words, words[1:])]
    pool = list(_LOW) + list(CITY_STATE)
    for w in cand:
        if len(w) >= 5 and w not in pool:
            hit = difflib.get_close_matches(w, pool, n=1, cutoff=0.84)
            if hit:
                return _LOW.get(hit[0]) or CITY_STATE[hit[0]]
    return None
