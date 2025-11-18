**Semantic Temporal Modelling of Wearable Device Data**

**Overview**

This repository provides an end-to-end pipeline that converts wearable device data including geographic files into a COPE-aligned (namespace: https://purl.archive.org/cope#) knowledge graph. The system performs data ingestion, temporal alignment, semantic modelling, analytics, and visualisation. The goal is to enable researchers, clinicians, and data scientists to study personal health trajectories using interoperable semantic standards.

**Key Features**

1. Data Ingestion and Normalisation

    - Parses Apple Health export.xml, including records, workouts, and metadata.
    - Loads GPX files containing timestamped GPS routes.
    - Normalises both into a unified internal format.

2. Temporal Matching and Linkage

    - Matches workouts with GPX traces based on overlapping timestamps.
    - Links Apple Health records (e.g., heart rate, HRV, steps) to specific workouts using a configurable time window (e.g., ±300 seconds).
    - Enables precise contextualisation of physiological signals.

3. COPE Semantic Mapping

    All data is mapped to the COPE ontology, generating RDF triples such as:
    
    <https://example.org/cope/record/858583> a cope:Observation ;
        cope:observedProperty "HKQuantityTypeIdentifierStepCount" ;
        cope:time "2021-03-24T07:56:12+09:30"^^xsd:dateTime ;
        cope:unit "count" ;
        cope:value "952" .

    The KG uses:
        cope:Observation for health metrics
        cope:Activity for workouts
        cope:Trajectory for GPX data
        cope:Record to represent Apple Health entries

4. Analytics Layer
    The knowledge graph supports advanced analytics, including:
        - HRV and heart rate trends
        - Activity and workout summaries
        - Apple Watch fitness metrics
        - Anomaly detection through statistical or semantic methods

5. Visualisation Tools

    The repository includes:
    - Knowledge graph interactive visualisations
    - Neo4j + NeoDash dashboards
    - GPX trajectory plots


**Example Repository Structure**

After cloning this project, you only need to place your export.xml and workout-routes folder inside this project folder as follows.

├── export.xml
│  
├── workout-routes/
│   ├── route_2021-03-10_9.05am.gpx
│   
├── pipeline.py
│   
└── README.md

**System Workflow**

The processing pipeline follows:

Apple Health XML + GPX
        ↓
Parsing and Normalisation
        ↓
Temporal Matching (Workouts ↔ Records ↔ GPX)
        ↓
COPE Semantic Modelling
        ↓
RDF Knowledge Graph Export
        ↓
Analytics and Visualisation


**Installation**

git clone https://github.com/AsaraSenaratne/apple-health-data-modelling.git
cd <repo-name>
pip install -r requirements.txt

**Usage**

Run full pipeline
python run_pipeline.py --xml data/export.xml --gpx data/gpx/


**Citation**

If this repository supports academic work, please cite:

Senaratne, Asara.;Seneviratne, Leelanga (2025). Semantic Modelling of Apple Health Data Using COPE (Chronic Observation and Progression Events) Ontology.

**Contributing**

Contributions, improvements, and issues are welcome. Please submit a pull request or open an issue to discuss changes.
