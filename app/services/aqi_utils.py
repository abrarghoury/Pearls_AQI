from config.constants import AQI_THRESHOLDS

# =====================================================
# AQI CLASS LABELS & COLORS
# =====================================================

AQI_CLASS_LABELS = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy (Sensitive)",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous"
}

AQI_CLASS_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
    4: "#8e44ad",
    5: "#7f0000"
}

AQI_CLASS_RANGES = {
    0: "0 – 50",
    1: "51 – 100",
    2: "101 – 150",
    3: "151 – 200",
    4: "201 – 300",
    5: "301+"
}


# =====================================================
# AQI VALUE → CATEGORY
# =====================================================
def aqi_category(aqi: float) -> str:
    if aqi is None:
        return "Unknown"

    try:
        aqi = float(aqi)
    except:
        return "Unknown"

    if aqi <= AQI_THRESHOLDS["good"]:
        return "Good"
    elif aqi <= AQI_THRESHOLDS["moderate"]:
        return "Moderate"
    elif aqi <= AQI_THRESHOLDS["unhealthy_sensitive"]:
        return "Unhealthy (Sensitive)"
    elif aqi <= AQI_THRESHOLDS["unhealthy"]:
        return "Unhealthy"
    elif aqi <= AQI_THRESHOLDS["very_unhealthy"]:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# =====================================================
# AQI VALUE → COLOR
# =====================================================
def aqi_color_from_value(aqi: float) -> str:
    if aqi is None:
        return "#95a5a6"

    try:
        aqi = float(aqi)
    except:
        return "#95a5a6"

    if aqi <= AQI_THRESHOLDS["good"]:
        return "#2ecc71"
    elif aqi <= AQI_THRESHOLDS["moderate"]:
        return "#f1c40f"
    elif aqi <= AQI_THRESHOLDS["unhealthy_sensitive"]:
        return "#e67e22"
    elif aqi <= AQI_THRESHOLDS["unhealthy"]:
        return "#e74c3c"
    elif aqi <= AQI_THRESHOLDS["very_unhealthy"]:
        return "#8e44ad"
    else:
        return "#7f0000"


# =====================================================
# AQI CLASS → LABEL
# =====================================================
def aqi_class_label(aqi_class: int) -> str:
    if aqi_class is None:
        return "Unknown"

    try:
        aqi_class = int(aqi_class)
    except:
        return "Unknown"

    return AQI_CLASS_LABELS.get(aqi_class, "Unknown")


# =====================================================
# AQI CLASS → COLOR
# =====================================================
def aqi_class_color(aqi_class: int) -> str:
    if aqi_class is None:
        return "#95a5a6"

    try:
        aqi_class = int(aqi_class)
    except:
        return "#95a5a6"

    return AQI_CLASS_COLORS.get(aqi_class, "#95a5a6")


# =====================================================
# AQI CLASS → FULL INFO (label + range)
# =====================================================
def aqi_class_info(aqi_class: int):
    if aqi_class is None:
        return {"label": "Unknown", "range": "N/A"}

    try:
        aqi_class = int(aqi_class)
    except:
        return {"label": "Unknown", "range": "N/A"}

    return {
        "label": AQI_CLASS_LABELS.get(aqi_class, "Unknown"),
        "range": AQI_CLASS_RANGES.get(aqi_class, "N/A")
    }
