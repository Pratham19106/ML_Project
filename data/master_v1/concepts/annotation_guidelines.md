# Traffic Sign Concept Annotation Guidelines

## 1. Overview

These guidelines govern the formal assignment of concept vectors to the 43 traffic-sign classes in GTSRB for the Master Dataset (`data/master_v1/concepts/class_concepts.csv`).

Because this study uses **class-level concept attribution** verified across representative clean samples, every sample belonging to class $k$ inherits the canonical concept vector of class $k$. The purpose of this guideline is to ensure that the canonical vector is objective, unambiguous, and traceable to traffic-sign specifications.

---

## 2. Core Annotation Rules

### 2.1 Shape Assignment
Assign the shape of the outer sign contour:
- If the sign is circular (regulatory prohibitions, speed limits, mandatory arrows), assign `circle`.
- If the sign is triangular pointing upward (danger warnings), assign `triangle_up`.
- If the sign is triangular pointing downward (Yield), assign `triangle_down`.
- If the sign is an octagon (Stop), assign `octagon`.
- If the sign is a tilted square / lozenge (Priority road), assign `diamond`.
- Note: Mounting plates or poles are not part of the sign shape.

### 2.2 Colour Profile Assignment
Assign the dominant interior field coloration:
- White background with red perimeter: `white_red` (e.g., standard speed limits, warning signs, no passing).
- Solid blue background with white graphics: `blue_white` (e.g., mandatory turn arrows, roundabout).
- White background with black markings and diagonal slash: `white_black` (e.g., cancellation of speed limits or passing restrictions).
- Solid red field with horizontal white bar: `red_white` (e.g., No Entry, StVO 267).
- Yellow diamond center with white outer border: `yellow_white` (e.g., Priority road, StVO 306).

### 2.3 Border Assignment
- Signs surrounded by a red outer ring/triangle: `red_border`.
- Signs with a contrasting white outer line: `white_border`.
- Signs with a black outer border or cancellation bar: `black_border`.
- Signs with no distinct framing border: `none`.

### 2.4 Symbol Category & Symbol Content Assignment
- Every class must have both:
  1. A coarse functional family: `symbol_category` (e.g., `speed_numeral`, `warning_hazard`, `mandatory_direction`, `prohibition_restriction`, `priority_geometry`, `deregulation_cancel`).
  2. A fine-grained semantic identity: `symbol_content` (e.g., `30`, `curve_left`, `pedestrians`, `turn_right`).
- **Rule of Non-Redundancy:** `symbol_content` must uniquely distinguish classes within the same coarse family so that an Oracle CBM can recover $100\%$ of classes from ground-truth concepts without ambiguity.

---

## 3. Handling Ambiguous and Edge Cases

1. **End of Speed / Passing Limits (Classes 6, 32, 41, 42):**
   - Shape: `circle`.
   - Colour profile: `white_black`.
   - Border: `black_border`.
   - Symbol category: `deregulation_cancel`.
   - Symbol content: `end_speed_80`, `end_all_restrictions`, `end_no_passing_cars`, `end_no_passing_trucks`.

2. **No Vehicles (Class 15):**
   - White circle with red border, completely empty inside.
   - Symbol category: `none`.
   - Symbol content: `empty_field`.

3. **General Caution (Class 18):**
   - Warning triangle containing an exclamation mark.
   - Symbol category: `warning_hazard`.
   - Symbol content: `general_caution` (or `exclamation_mark`).

---

## 4. Quality Control & Audit Protocol

1. **Sample Verification:** For each class, inspect at least 5 clean, high-resolution images from different tracks to ensure the canonical vector matches all visual manifestations.
2. **Provenance Fields:**
   - `vienna_reference`: The specific Vienna Convention section/sign code (e.g., `C, 14` for Speed limit, `B, 2a` for STOP).
   - `stvo_reference`: The official German StVO / VzKat sign number (e.g., `Zeichen 274` for Speed limit, `Zeichen 206` for STOP).
   - `annotator`: Identifying tag of the researcher or agent who mapped the entry.
   - `annotation_confidence`: `high` (standard unambiguous sign), `medium` (variant in German implementation).

