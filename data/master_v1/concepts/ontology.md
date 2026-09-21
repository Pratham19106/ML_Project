# Traffic Sign Concept Ontology Specification (Version 1.0)

## 1. Purpose and Theoretical Grounding

This document formalizes the machine-readable concept ontology used by the Concept Bottleneck Models (CBM) in this research.

As required by [Structured_Research_Proposal.md](file:///d:/ML_Project/Structured_Research_Proposal.md) and [AGENT_PROJECT_CONTEXT.md](file:///d:/ML_Project/AGENT_PROJECT_CONTEXT.md):
- Concepts are strictly **grounded in and derived from** international and national standards:
  - **Vienna Convention on Road Signs and Signals (1968)**: establishes global structural sign families, shapes, color conventions, and symbol classes.
  - **German Straßenverkehrs-Ordnung (StVO / VzKat)**: governs the specific national implementation of GTSRB signs (e.g., German Zeichen numbers, glyph styles, and color profiles).
- The vocabulary does **not** claim to be legally mandated in its exact ML encoding; rather, it forms an objective, standards-traceable semantic bottleneck that separates visual perception from class reasoning.

---

## 2. Hierarchical Concept Axes

To ensure the concept vector is **sufficiently expressive to uniquely distinguish all 43 GTSRB classes** (enabling a meaningful Oracle CBM) while preserving **broad functional interpretability**, the symbol representation is structured hierarchically into `symbol_category` and `symbol_content`.

### Axis 1: Shape (`shape`)
Physical geometric silhouette of the sign plate:
* `circle`: Circular signs (regulatory: prohibitions, speed limits, mandatory commands).
* `triangle_up`: Equilateral triangle pointing upward (danger warning signs, Vienna Class A).
* `triangle_down`: Equilateral triangle pointing downward (Yield sign, Vienna B.1 / StVO 205).
* `octagon`: Regular eight-sided polygon (STOP sign, Vienna B.2a / StVO 206).
* `diamond`: Square tilted 45° with a horizontal diagonal (Priority Road, Vienna B.3 / StVO 306).
* `rectangle`: Upright or horizontal rectangle (informational signs).

### Axis 2: Colour Profile (`colour_profile`)
Primary interior background and field coloration:
* `white_red`: White background with red border / red elements.
* `blue_white`: Solid blue background with white symbols/arrows (Mandatory signs, Vienna Class D).
* `yellow_white`: Yellow center field with white/black outer frame (Priority signs, StVO 306).
* `white_black`: White background with black markings or black strike-through bars (De-restriction signs).
* `red_white`: Dominant solid red field with white symbol (e.g., No Entry, StVO 267).

### Axis 3: Border Type (`border_type`)
Presence and color of the outer framing border:
* `red_border`: Standard circular or triangular red border.
* `black_border`: Thin black outline or cancellation bar.
* `white_border`: Outer white framing rim (common on blue circular mandatory signs and stop signs).
* `none`: No distinct border.

### Axis 4: Symbol Category (`symbol_category`) — Broad Functional Family
The functional semantic category of the glyph:
* `speed_numeral`: Numeric speed limit representation.
* `warning_hazard`: Pictogram indicating a roadway hazard or obstacle ahead.
* `mandatory_direction`: Prescribed directional arrow or movement indicator.
* `prohibition_restriction`: Restriction or exclusion symbol (prohibiting vehicle types or maneuvers).
* `priority_geometry`: Geometric right-of-way assignment indicator.
* `deregulation_cancel`: Diagonal cancellation bar indicating end of restrictions.
* `none`: Blank interior field.

### Axis 5: Symbol Content (`symbol_content`) — Fine-Grained Identity
The specific semantic glyph or content essential for class-level disambiguation in an Oracle CBM:
* **Speed limits:** `20`, `30`, `50`, `60`, `70`, `80`, `100`, `120`
* **Warnings / Hazards:** `bumpy_road`, `slippery_road`, `curve_left`, `curve_right`, `double_curve`, `road_narrows_right`, `road_work`, `traffic_light`, `pedestrians`, `children`, `bicycles`, `ice_snow`, `wild_animals`, `general_caution`
* **Mandatory directions:** `turn_right`, `turn_left`, `ahead_only`, `straight_or_right`, `straight_or_left`, `keep_right`, `keep_left`, `roundabout`
* **Prohibitions & Restrictions:** `no_entry_bar`, `no_vehicles`, `truck_restriction`, `no_passing_cars`, `no_passing_trucks`
* **Priority:** `crossroad_hazard`, `priority_diamond`, `yield_inverted_triangle`, `stop_text`
* **Cancellations:** `end_speed_80`, `end_all_restrictions`, `end_no_passing_cars`, `end_no_passing_trucks`

---

## 3. Semantic Consistency Constraints

The ontology enables explicit rule-based validation checks (Consistency Layer):
1. **Octagon Constraint:** `shape == octagon` $\iff$ `class == 14 (Stop)`.
2. **Inverted Triangle Constraint:** `shape == triangle_down` $\iff$ `class == 13 (Yield)`.
3. **Diamond Constraint:** `shape == diamond` $\iff$ `class == 12 (Priority road)`.
4. **Mandatory Direction Constraint:** `shape == circle` AND `colour_profile == blue_white` $\implies$ `class in [33..40]`.
5. **Speed Limit Constraint:** `shape == circle` AND `border_type == red_border` AND `symbol_category == speed_numeral` $\implies$ `class in [0, 1, 2, 3, 4, 5, 7, 8]`.

---

## 4. Verification and Freezing Process

Before any model training begins:
1. Every class row in `class_concepts.csv` must be populated with verified references to Vienna Convention Annex 1 and German StVO / VzKat.
2. A random audit sample of clean images per class will be visually checked against the proposed concept vector.
3. The resulting CSV will be checksummed and frozen as `class_concepts_v1.csv`.

