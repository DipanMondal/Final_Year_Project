def city_key(city: str, country_code: str | None = None):
    c = city.strip().lower().replace(" ", "_")
    cc = (country_code or "").strip().lower()
    return f"{c}_{cc}" if cc else c
